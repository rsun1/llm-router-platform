import streamlit as st
import requests
import time 
import plotly.graph_objects as go 
import io 
import pandas as pd 
import os
import json
from datetime import datetime, timedelta
import plotly.express as px
import yaml

datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
configpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')

@st.cache_data
def load_data():
    log = os.path.join(datapath, 'logs.csv')
    data = pd.DataFrame(pd.read_csv(log))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data 

def load_config():
    config = os.path.join(configpath, 'config.yaml')
    with open(config) as f :
        return yaml.safe_load(f)
    
    
st.set_page_config(
    layout= 'wide',
    page_title='LLM Router',
    initial_sidebar_state='expanded'   
)

def sidebar():
    with st.sidebar:
        st.header('Navigation')
        
        
        selected_page = st.radio(
            'Navigation',
            ['User Requests','Overview', 'Models','Performance','Users','Costs', 'Alerts', 'Logs']
        )
        return selected_page 
    
    st.markdown("-----")
    st.write("System Status: Operational")
    st.write("Uptime: 99.9%")
    st.write("Active Models: 4/4")
    st.write("Version: v1.1")    
    

data_logs = load_data()
def compute_overview_metrics(data_logs):
    total_requests = data_logs["query_id"].nunique()
    avg_response = data_logs["latency_ms"].mean()
    success_rate = (data_logs["status"]=="success").mean()*100.00
    daily_cost = data_logs["cost_usd"].sum()
    cache_hit_rate = (data_logs["cached_response"]==True).mean()*100.00
    
    return total_requests, avg_response, success_rate, daily_cost, cache_hit_rate

def compute_hourly_stats(data_logs):
    data_logs["hour"]=data_logs["timestamp"].dt.hour
    hourly_requests=data_logs.groupby('hour').size().reset_index(name='requests')
    hourly_latency=data_logs.groupby('hour')['latency_ms'].mean().reset_index()
    return hourly_latency, hourly_requests

def compute_distribution_stats(data_logs):
    query_dist = data_logs['query_type'].value_counts().reset_index()
    cost_breakdown = data_logs.groupby('selected_model')['cost_usd'].sum().reset_index()
    user_tiers = data_logs['user_tier'].value_counts().reset_index()
    return query_dist, cost_breakdown, user_tiers

def compute_model_stats(data_logs):
    model_stats = data_logs.groupby('selected_model').agg(
        Requests=('query_id', 'count'),
        SuccessRate=('status', lambda x : (x=='success').mean()*100.0),
        AvgLatency=('latency_ms', 'mean'),
        Cost=('cost_usd','sum'),
    ).reset_index()
    model_stats['Efficiency'] = '-'
    model_stats['Status'] = '-'
    model_stats=model_stats.rename(columns={'selected_model':'Model'})
    model_stats=model_stats[['Model','Status','Requests','SuccessRate','AvgLatency','Cost','Efficiency']] 
    return model_stats


def page_overview():
    st.header("System Overview")
    total_requests, avg_response, success_rate, daily_cost, cache_hit_rate = compute_overview_metrics(data_logs)

    cols = st.columns(5)
    cols[0].metric('Total Requests', total_requests)
    cols[1].metric('Avg Response Time', f"{avg_response:.0f}ms")
    cols[2].metric('Success Rate', f"{success_rate:.1f}%")
    cols[3].metric('Daily Cost', f"${daily_cost:.2f}")
    cols[4].metric('Cache Hit Rate', f"{cache_hit_rate:.1f}%") 
    
    hourly_latency, hourly_requests = compute_hourly_stats(data_logs)
    col_chart = st.columns(2)
    st.subheader('Performance Analytics')
    with col_chart[0]:
        chart_requests = px.area(hourly_requests, x='hour', y='requests', title='Request Volume')
        st.plotly_chart(chart_requests, use_container_width=True)
    with col_chart[1]:
        chart_latency = px.area(hourly_latency, x='hour', y='latency_ms' ,title='Response Times')
        st.plotly_chart(chart_latency, use_container_width=True)    
    
    st.subheader('Model Performance')
    model_stats = compute_model_stats(data_logs)
    st.dataframe(model_stats, 
                    hide_index=True,
                    column_config={
                        'AvgLatency': st.column_config.NumberColumn('Avg Latency', format="%.0f ms"),
                        'Cost': st.column_config.NumberColumn('Cost',format='$%.3f'),
                        'SuccessRate': st.column_config.NumberColumn('Success Rate',format='%.1f%%')
                    }
                    )
    
    st.subheader('Distribution Analytics')
    query_dist, cost_breakdown, user_tiers = compute_distribution_stats(data_logs)
    col_dist = st.columns(3)
    with col_dist[0]:
        fig= px.pie(query_dist, names='query_type', values='count', hole=0.5, title='Query Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_dist[1]:
        fig = px.pie(cost_breakdown, names='selected_model', values='cost_usd', hole=0.5, title='Cost Breakdown')
        st.plotly_chart(fig, use_container_width=True)

    with col_dist[2]:
        fig = px.pie(user_tiers, names='user_tier', values='count', hole=0.5, title='User Tiers')
        st.plotly_chart(fig, use_container_width=True)    
    
    
    
    
    
def page_models():
    st.header("Models")
    model_stats = compute_model_stats(data_logs)
    st.dataframe(model_stats, 
                 hide_index=True,
                 column_config={
                     'AvgLatency': st.column_config.NumberColumn('Avg Latency', format="%.0f ms"),
                     'Cost': st.column_config.NumberColumn('Cost',format='$%.3f'),
                     'SuccessRate': st.column_config.NumberColumn('Success Rate',format='%.1f%%')
                 }
                 )
   
    st.subheader('Distribution Analytics')
    query_dist, cost_breakdown, user_tiers = compute_distribution_stats(data_logs)
    col_dist = st.columns(3)
    with col_dist[0]:
        fig= px.pie(query_dist, names='query_type', values='count', hole=0.5, title='Query Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_dist[1]:
        fig = px.pie(cost_breakdown, names='selected_model', values='cost_usd', hole=0.5, title='Cost Breakdown')
        st.plotly_chart(fig, use_container_width=True)

    with col_dist[2]:
        fig = px.pie(user_tiers, names='user_tier', values='count', hole=0.5, title='User Tiers')
        st.plotly_chart(fig, use_container_width=True)    
 
def page_performance():
    st.header("Performance Analytics")

    hourly_latency, hourly_requests = compute_hourly_stats(data_logs)
    col_chart = st.columns(2)
    with col_chart[0]:
        chart_requests = px.area(hourly_requests, x='hour', y='requests', title='Request Volume')
        st.plotly_chart(chart_requests, use_container_width=True)
    with col_chart[1]:
        chart_latency = px.area(hourly_latency, x='hour', y='latency_ms' ,title='Response Times')
        st.plotly_chart(chart_latency, use_container_width=True)    


    total_requests, avg_response, success_rate, daily_cost, cache_hit_rate = compute_overview_metrics(data_logs)

    cols = st.columns(5)
    cols[0].metric('Total Requests', total_requests)
    cols[1].metric('Avg Response Time', f"{avg_response:.0f}ms")
    cols[2].metric('Success Rate', f"{success_rate:.1f}%")
    cols[3].metric('Daily Cost', f"${daily_cost:.2f}")
    cols[4].metric('Cache Hit Rate', f"{cache_hit_rate:.1f}%") 
    
def page_users():
    st.header("Users")
    st.subheader('Distribution Analytics')
    _, _, user_tiers = compute_distribution_stats(data_logs)
    
    fig = px.pie(user_tiers, names='user_tier', values='count', hole=0.5, title='User Tiers')
    st.plotly_chart(fig, use_container_width=True)    

def page_costs():
    st.header("Costs")     
    st.subheader('Distribution Analytics')
    _, cost_breakdown, _ = compute_distribution_stats(data_logs)
    
    fig = px.pie(cost_breakdown, names='selected_model', values='cost_usd', hole=0.5, title='Cost Breakdown')
    st.plotly_chart(fig, use_container_width=True)    
    
def page_alerts():
    st.header("System Alerts")
    st.success("System Health - All services operating within normal parameters")
    st.warning("High Latency Detected - Llama 3.1 70B showing elevated response time (2.1s avg)")
    
    st.subheader("Alert Configuration")
    row1 =  st.columns(2)
    cpu_threshold=row1[0].slider("CPU Alert Threshold (%)", 0, 100,80)
    error_threshold=row1[1].slider("Error Rate Threshold (%)", 0,100,5)

    row2 = st.columns(2)
    memory_threshold=row2[0].slider("Memory Alert Threshold (%)", 0,100,80)
    latency_threshold=row2[1].slider("Latency Threshold (ms)", 0, 10000, 2000) 
    
    if st.button("Update Alert Settings"):
        st.success("Alert settings updated successfully!")   
        
def page_logs():
    st.header("System Logs")
    row = st.columns(3)
    log_level=row[0].selectbox("Log Level", ["All","INFO","WARNING","ERROR"])
    component=row[1].selectbox("Component", ["All","inference","router","slack","pipeline"])
    num_logs=row[2].slider("Number of logs", 50,200,50)
    
    log_entries = [
        {"time": "14:25:32", "level": "INFO", "component": "inference", "content": "Model inference completed successfully", "request_id":
  "req-abc123"},
        {"time": "14:23:15", "level": "WARNING", "component": "router", "content": "High latency detected for model gpt-4-turbo",
  "request_id": "req-abc122"},
        {"time": "14:22:48", "level": "INFO", "component": "slack", "content": "Message processed for user user-789", "request_id":
  "req-abc121"},
        {"time": "14:20:12", "level": "ERROR", "component": "pipeline", "content": "Failed to insert batch to ClickHouse", "request_id":
  "req-abc120"},
    ]
    level_style = {"INFO": st.info, "WARNING": st.warning, "ERROR": st.error}
    for log in log_entries:
        level_style[log["level"]](f"{log['time']} [{log['level']}] {log['component']} — {log['content']}  \nRequest ID: {log['request_id']}")

def page_playground():
    st.header('User Requests')
    st.text('Call FastAPI via python -m uvicorn llm_router_part5_deploy:app --reload --port 8080')
    with st.form('User_Request'):
        query_text = st.text_area('Your question')
        user_tier = st.selectbox('User tier',['free','premium','enterprise'])
        user_id = st.text_input('User ID')
        submitted = st.form_submit_button('Submit')
    if submitted:
        sub_request = {
            'query_text': query_text,
            'user_tier':user_tier,
            'user_id':user_id,
        }
        with st.spinner('Routing your request...'):
            try:
                res = requests.post('http://127.0.0.1:8080/route',json=sub_request, timeout=180)
                if res.status_code == 200:
                    result = res.json() 
                    st.write(result)
                else:
                    st.error(f'Error: {res.status_code}, {res.text}')
            except requests.exceptions.RequestException as e:
                st.error(f'Failed to connect backend. Please start FastAPI first. Error: {e}')
                return


def main():
    page = sidebar()
    if page == 'User Requests':
        page_playground()
    elif page == 'Overview':
        page_overview()
    elif page == 'Models':
        page_models()
    elif page == 'Performance':
        page_performance()
    elif page == 'Users':
        page_users()
    elif page == 'Costs':
        page_costs()
    elif page == 'Alerts':
        page_alerts()
    elif page == 'Logs':
        page_logs()
        
if __name__ == '__main__':
    main()