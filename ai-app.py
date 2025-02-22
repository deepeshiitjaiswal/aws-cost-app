import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
import boto3
from dotenv import load_dotenv
try:
    import ollama
except ImportError:
    st.error("Ollama library not installed. Please run 'pip install ollama'")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AWS Cost Dashboard")

# Define function to load AWS credentials
def load_aws_credentials():
    """Load AWS credentials from environment variables using python-dotenv."""
    load_dotenv()  # Load variables from .env file if present
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION", "us-east-1")  # Default region if not set

    if not aws_access_key_id or not aws_secret_access_key:
        raise Exception("AWS credentials not found in environment variables.")
    
    return aws_access_key_id, aws_secret_access_key, region_name

# Cache data summary generation for 1 hour
@st.cache_data(ttl=3600)
def generate_data_summary(df, service_cost, total_cost, selected_region, start_date, end_date):
    """Optimized data summary generation using vectorization"""
    logger.info("Generating data summary for AI analysis")
    try:
        # Vectorized string formatting for service costs
        service_summary = service_cost.assign(
            Cost=lambda x: x['Cost'].apply(lambda v: f"${v:,.2f}")
        ).to_string(index=False, header=False)
        
        summary = (
            f"AWS Cost Data Summary ({start_date} to {end_date}):\n"
            f"Total Cost: ${total_cost:,.2f}\n"
            f"Selected Region: {selected_region}\n\n"
            f"Service Costs:\n{service_summary}"
        )
        
        if selected_region == 'All':
            regions = df['Region'].unique()
            summary += f"\n\nRegions with Costs: {', '.join(regions)}"
            
        logger.info("Data summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        st.error("Failed to generate data summary. Check logs for details.")
        raise

# Cache AI responses for similar queries
@st.cache_data(ttl=600, show_spinner=False)
def get_ai_response(system_prompt, user_prompt):
    """Get cached AI response with performance optimizations"""
    try:
        response = ollama.chat(
            model='mistral:7b-instruct',  # Use smaller/faster model
            options={
                'temperature': 0.3,      # More deterministic responses
                'num_predict': 256       # Limit response length
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def get_daily_cost_data(client, start_date, end_date):
    """
    Fetch daily cost data from AWS Cost Explorer in real time.
    This function groups data by SERVICE and REGION.
    """
    try:
        response = client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                {'Type': 'DIMENSION', 'Key': 'REGION'}
            ]
        )
        
        cost_data = []
        # Process the response to extract cost details
        for result in response.get("ResultsByTime", []):
            date = result["TimePeriod"]["Start"]
            for group in result.get("Groups", []):
                keys = group.get("Keys", [])
                # Expecting first key to be Service and second to be Region
                if len(keys) >= 2:
                    service = keys[0]
                    region = keys[1]
                else:
                    service = keys[0] if keys else "Unknown"
                    region = "Unknown"
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                cost_data.append({
                    "Date": date,
                    "Service": service,
                    "Cost": amount,
                    "Region": region
                })
        return cost_data
    except Exception as e:
        logger.error(f"Error fetching cost data: {str(e)}")
        st.error("Error fetching cost data from AWS Cost Explorer.")
        return []

def create_dashboard():
    """Main dashboard creation function"""
    logger.info("Initializing AWS Cost Dashboard")
    
    try:
        # Load AWS credentials and create client
        aws_creds = load_aws_credentials()
        client = boto3.client(
            'ce',
            aws_access_key_id=aws_creds[0],
            aws_secret_access_key=aws_creds[1],
            region_name=aws_creds[2]
        )

        # Set date range (Last 30 days)
        today = datetime.today().date()
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        logger.info(f"Date range set to {start_date} - {end_date}")

        # Fetch and process real-time cost data
        cost_data = get_daily_cost_data(client, start_date, end_date)
        if not cost_data:
            logger.warning("No cost data found for the date range")
            st.error("No cost data found for the given date range.")
            return

        df = pd.DataFrame(cost_data)
        all_regions = sorted(df['Region'].unique().tolist())
        region_options = ['All'] + all_regions
        selected_region = st.selectbox("Select AWS Region to Filter:", region_options, index=0)

        if selected_region != 'All':
            df = df[df['Region'] == selected_region]

        service_cost = df.groupby('Service', as_index=False)['Cost'].sum().sort_values(by='Cost', ascending=False)
        total_cost = service_cost['Cost'].sum()
        logger.info(f"Total calculated cost: ${total_cost:,.2f}")

        # Main dashboard layout
        st.title("AWS Cost Dashboard with AI Analysis")
        st.markdown(f"**Date Range:** {start_date} to {end_date}")

        if selected_region == 'All':
            st.markdown("**Regions:** All AWS Regions")
        else:
            st.markdown(f"**Region:** {selected_region}")

        st.metric(label="Total AWS Spend (Last 30 Days)", value=f"${total_cost:,.2f}")

        # Visualization section
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(service_cost, values='Cost', names='Service', 
                            title="Cost Distribution by Service", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(df, x='Date', y='Cost', color='Service',
                            title="Daily AWS Costs by Service", barmode='stack')
            fig_bar.update_layout(xaxis=dict(type='category'))
            st.plotly_chart(fig_bar, use_container_width=True)

        # AI Chat Section
        logger.info("Initializing AI Chat section")
        st.divider()
        st.subheader("AI Cost Analyst (Powered by Ollama)")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input for AI analysis
        if prompt := st.chat_input("Ask about your AWS costs..."):
            logger.info(f"User query: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                with st.spinner("Analyzing your query..."):
                    # Generate data summary once per session
                    if "data_summary" not in st.session_state:
                        st.session_state.data_summary = generate_data_summary(
                            df, service_cost, total_cost, 
                            selected_region, start_date, end_date
                        )
                    
                    system_prompt = (
                        "You are an AWS cost analyst. Provide concise, professional answers "
                        f"using this data: {st.session_state.data_summary}"
                    )
                    
                    # Get cached AI response
                    ai_response = get_ai_response(system_prompt, prompt)
                
                logger.info("Successfully generated AI response")
                
            except Exception as e:
                ai_response = f"Error: {str(e)}"
                logger.error(f"AI response error: {str(e)}")

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)

        # Clear chat button
        if st.button("Clear Chat History"):
            logger.info("Clearing chat history")
            st.session_state.messages = []
            st.experimental_rerun()

    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        st.error("A critical error occurred. Check logs for details.")
        raise

def main():
    try:
        logger.info("Starting application")
        create_dashboard()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        st.error("The application has encountered a critical error and needs to stop.")

if __name__ == "__main__":
    main()
