import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
import boto3
from dotenv import load_dotenv

def load_aws_credentials():
    # Load environment variables from .env file
    load_dotenv()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION", "us-east-1")
    return aws_access_key_id, aws_secret_access_key, region_name

def get_daily_cost_data(client, start_date, end_date):
    """
    Retrieve daily cost data from AWS Cost Explorer, grouped by both SERVICE and REGION.
    Returns a list of dictionaries with columns: Date, Service, Region, and Cost.
    """
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'},
            {'Type': 'DIMENSION', 'Key': 'REGION'}
        ]
    )
    
    results = []
    for day_item in response['ResultsByTime']:
        date_str = day_item['TimePeriod']['Start']
        for group in day_item.get('Groups', []):
            # group['Keys'] will contain [ServiceName, RegionName]
            service = group['Keys'][0]
            region = group['Keys'][1]
            cost_amount = float(group['Metrics']['UnblendedCost']['Amount'])
            results.append({
                'Date': date_str,
                'Service': service,
                'Region': region,
                'Cost': cost_amount
            })
    return results

def create_dashboard():
    # 1. Load AWS credentials
    aws_access_key_id, aws_secret_access_key, region_name = load_aws_credentials()

    # 2. Initialize Cost Explorer client
    client = boto3.client(
        'ce',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    # 3. Define date range (last 30 days)
    today = datetime.today().date()
    start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    # 4. Get daily cost data
    cost_data = get_daily_cost_data(client, start_date, end_date)
    if not cost_data:
        st.error("No cost data found for the given date range.")
        return

    # Convert to pandas DataFrame
    df = pd.DataFrame(cost_data)

    # --- Region Filter ---
    # Let user pick a specific region or view all
    all_regions = sorted(df['Region'].unique().tolist())
    region_options = ['All'] + all_regions
    selected_region = st.selectbox("Select AWS Region to Filter:", region_options, index=0)

    # Filter the DataFrame if a specific region is chosen
    if selected_region != 'All':
        df = df[df['Region'] == selected_region]

    # 5. Summaries
    #    a) Total cost by service (for the filtered region(s))
    service_cost = df.groupby('Service', as_index=False)['Cost'].sum().sort_values(by='Cost', ascending=False)
    #    b) Overall total cost
    total_cost = service_cost['Cost'].sum()

    # 6. Build Streamlit Dashboard
    st.title("AWS Cost Dashboard (Last 30 Days)")
    st.markdown(f"**Date Range:** {start_date} to {end_date}")

    if selected_region == 'All':
        st.markdown("**Regions:** All AWS Regions")
    else:
        st.markdown(f"**Region:** {selected_region}")

    # Show total cost in US dollars
    st.metric(label="Total AWS Spend (Last 30 Days)", value=f"${total_cost:,.2f}")

    # --- CHART 1: Donut Chart of Cost by Service ---
    fig_pie = px.pie(
        service_cost,
        values='Cost',
        names='Service',
        title="Cost Distribution by Service",
        hole=0.4  # Creates the "donut" hole
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- CHART 2: Bar/Column Chart of Daily Costs by Service ---
    # We'll plot from the filtered df so it reflects the chosen region
    fig_bar = px.bar(
        df,
        x='Date',
        y='Cost',
        color='Service',
        title="Daily AWS Costs by Service",
        barmode='stack'
    )
    fig_bar.update_layout(xaxis=dict(type='category'))  # ensures daily labels are discrete
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Detailed Table of Costs by Service ---
    st.subheader("Detailed Cost by Service")
    st.dataframe(service_cost.reset_index(drop=True))

    # Optional: Provide simple suggestions
    st.subheader("Cost Optimization Suggestions")
    cost_threshold = 50.0  # Example threshold
    high_spend_services = service_cost[service_cost['Cost'] > cost_threshold]
    if not high_spend_services.empty:
        for _, row in high_spend_services.iterrows():
            st.write(
                f"- **{row['Service']}** spent **${row['Cost']:.2f}**. "
                "Consider reviewing usage, rightsizing resources, or exploring reserved instances/savings plans."
            )
    else:
        st.write("No high-cost services detected above threshold.")

def main():
    create_dashboard()

if __name__ == "__main__":
    main()
