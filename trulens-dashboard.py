import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import numpy as np
from snowflake.snowpark import Session
import requests

# Page configuration
st.set_page_config(
    page_title="AI Observability with TruLens",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Snowflake connection parameters
HOST = st.secrets["connections"]["snowflake"]["host"]
DATABASE = st.secrets["connections"]["snowflake"]["database"]
SCHEMA = st.secrets["connections"]["snowflake"]["schema"]
STAGE = st.secrets["connections"]["snowflake"]["stage"]
FILE = st.secrets["connections"]["snowflake"]["semantic_context_file"]

SNOWFLAKE_ACCOUNT_URL = f"https://{HOST}"
SNOWFLAKE_PAT = st.secrets["connections"]["snowflake"]["pat"]
API_ENDPOINT = "/api/v2/cortex/analyst/message"

# Create a single session that will be reused everywhere
@st.cache_resource
def create_snowpark_session():
    """Create a single Snowpark session to be shared across the application"""
    connection_parameters = {
        "account": st.secrets["connections"]["snowflake"]["account"],
        "user": st.secrets["connections"]["snowflake"]["user"],
        "password": st.secrets["connections"]["snowflake"]["password"],
        "role": st.secrets["connections"]["snowflake"].get("role"),
        "warehouse": st.secrets["connections"]["snowflake"].get("warehouse"),
        "database": DATABASE,
        "schema": SCHEMA
    }
    
    # Remove None values
    connection_parameters = {k: v for k, v in connection_parameters.items() if v is not None}
    
    try:
        session = Session.builder.configs(connection_parameters).create()
        return session
    except Exception as e:
        st.error(f"❌ Failed to connect to Snowflake: {str(e)}")
        return None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 0.3; }
        100% { transform: scale(0.8); opacity: 0.5; }
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .model-comparison-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f2937;
        text-align: center;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

def safe_json_parse(json_str):
    """Safely parse JSON strings"""
    try:
        if pd.isna(json_str) or json_str == "None":
            return {}
        return json.loads(json_str)
    except:
        return {}

@st.cache_data
def load_and_process_data():
    """Load and process all TruLens data from Snowflake"""
    
    # Get Snowpark session
    session = create_snowpark_session()
    if session is None:
        st.error("Cannot proceed without Snowflake connection")
        return None, None, None, None
    
    try:
        # Load data from Snowflake tables using the correct database and schema
        with st.spinner(f"Loading data from {DATABASE}.{SCHEMA}..."):
            # Load apps data
            apps_query = f"SELECT * FROM {DATABASE}.{SCHEMA}.TRULENS_APPS"
            apps_df = session.sql(apps_query).to_pandas()
            
            # Load feedback definitions
            feedback_defs_query = f"SELECT * FROM {DATABASE}.{SCHEMA}.TRULENS_FEEDBACK_DEFS"
            feedback_defs_df = session.sql(feedback_defs_query).to_pandas()
            
            # Load feedback results
            feedbacks_query = f"SELECT * FROM {DATABASE}.{SCHEMA}.TRULENS_FEEDBACKS"
            feedbacks_df = session.sql(feedbacks_query).to_pandas()
            
            # Load records
            records_query = f"SELECT * FROM {DATABASE}.{SCHEMA}.TRULENS_RECORDS"
            records_df = session.sql(records_query).to_pandas()
        
    except Exception as e:
        st.error(f"❌ Error loading data from Snowflake tables in {DATABASE}.{SCHEMA}: {str(e)}")
        st.error("Please ensure:")
        st.error("1. The Snowflake connection parameters are correct in secrets.toml")
        st.error("2. The TruLens tables exist in the specified database and schema")
        st.error("3. Your user has SELECT permissions on these tables")
        return None, None, None, None
    
    # Process apps data
    apps_processed = []
    for _, row in apps_df.iterrows():
        app_json = safe_json_parse(row['APP_JSON'])
        apps_processed.append({
            'APP_ID': row['APP_ID'],
            'APP_NAME': row['APP_NAME'],
            'MODEL': row['APP_VERSION'],
            'METADATA': app_json.get('metadata', {}),
            'APP_TYPE': app_json.get('metadata', {}).get('app_type', 'unknown')
        })
    apps_processed_df = pd.DataFrame(apps_processed)
    
    # Process feedback definitions
    feedback_defs_processed = []
    for _, row in feedback_defs_df.iterrows():
        feedback_json = safe_json_parse(row['FEEDBACK_JSON'])
        feedback_defs_processed.append({
            'FEEDBACK_DEFINITION_ID': row['FEEDBACK_DEFINITION_ID'],
            'NAME': feedback_json.get('supplied_name', 'Unknown'),
            'HIGHER_IS_BETTER': feedback_json.get('higher_is_better', True)
        })
    feedback_defs_processed_df = pd.DataFrame(feedback_defs_processed)
    
    # Process feedback results
    feedbacks_processed = []
    for _, row in feedbacks_df.iterrows():
        cost_json = safe_json_parse(row['COST_JSON'])
        calls_json = safe_json_parse(row['CALLS_JSON'])
        
        reasoning = ""
        if calls_json.get('calls') and len(calls_json['calls']) > 0:
            reasoning = calls_json['calls'][0].get('meta', {}).get('reason', '')
        
        feedbacks_processed.append({
            'FEEDBACK_RESULT_ID': row['FEEDBACK_RESULT_ID'],
            'RECORD_ID': row['RECORD_ID'],
            'FEEDBACK_DEFINITION_ID': row['FEEDBACK_DEFINITION_ID'],
            'RESULT': row['RESULT'],
            'STATUS': row['STATUS'],
            'NAME': row['NAME'],
            'COST': cost_json.get('cost', 0),
            'TOKENS': cost_json.get('n_tokens', 0),
            'REASONING': reasoning,
            'TIMESTAMP': pd.to_datetime(row['LAST_TS'], unit='s')
        })
    feedbacks_processed_df = pd.DataFrame(feedbacks_processed)
    
    # Process records
    records_processed = []
    for _, row in records_df.iterrows():
        record_json = safe_json_parse(row['RECORD_JSON'])
        cost_json = safe_json_parse(row['COST_JSON'])
        perf_json = safe_json_parse(row['PERF_JSON'])
        
        # Calculate latency
        latency = 0
        if perf_json.get('start_time') and perf_json.get('end_time'):
            try:
                start = pd.to_datetime(perf_json['start_time'])
                end = pd.to_datetime(perf_json['end_time'])
                latency = (end - start).total_seconds()
            except:
                latency = 0
        
        records_processed.append({
            'RECORD_ID': row['RECORD_ID'],
            'APP_ID': row['APP_ID'],
            'INPUT': row['INPUT'].strip('"') if pd.notna(row['INPUT']) else '',
            'OUTPUT': row['OUTPUT'].strip('"') if pd.notna(row['OUTPUT']) else '',
            'LATENCY': latency,
            'COST': cost_json.get('cost', record_json.get('cost', {}).get('cost', 0)),
            'TOKENS': cost_json.get('n_tokens', record_json.get('cost', {}).get('n_tokens', 0)),
            'TIMESTAMP': pd.to_datetime(row['TS'], unit='s'),
            'META': record_json.get('meta', {}),
            'COMPLEXITY': 'Simple'  # Default - can be enhanced based on input analysis
        })
    records_processed_df = pd.DataFrame(records_processed)
    
    # Add complexity classification based on input length and content
    if not records_processed_df.empty:
        records_processed_df['INPUT_LENGTH'] = records_processed_df['INPUT'].str.len()
        records_processed_df['COMPLEXITY'] = pd.cut(
            records_processed_df['INPUT_LENGTH'],
            bins=[0, 50, 150, float('inf')],
            labels=['Simple', 'Intermediate', 'Complex']
        )
    
    return apps_processed_df, feedback_defs_processed_df, feedbacks_processed_df, records_processed_df

# Load data
apps_df, feedback_defs_df, feedbacks_df, records_df = load_and_process_data()

if apps_df is None:
    st.stop()

# Merge data for analysis
if not feedbacks_df.empty and not records_df.empty and not apps_df.empty:
    # Merge feedbacks with records to get app info
    analysis_df = feedbacks_df.merge(records_df[['RECORD_ID', 'APP_ID', 'LATENCY', 'COMPLEXITY', 'INPUT', 'OUTPUT']], 
                                   on='RECORD_ID', how='left')
    analysis_df = analysis_df.merge(apps_df[['APP_ID', 'MODEL', 'APP_NAME']], on='APP_ID', how='left')
else:
    analysis_df = pd.DataFrame()

# Header
st.markdown('<div class="main-header">🤖 AI Observability with TruLens</div>', 
           unsafe_allow_html=True)

# Sidebar for filters and controls
st.sidebar.header("🎛️ Dashboard Controls")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Connection status (silent check)
session = create_snowpark_session()

# Data freshness and connection info
if not records_df.empty:
    latest_timestamp = records_df['TIMESTAMP'].max()
    st.sidebar.info(f"📅 Latest: {latest_timestamp.strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.info(f"🗂️ {DATABASE}.{SCHEMA}")

# Date range filter
if not records_df.empty:
    min_date = records_df['TIMESTAMP'].min().date()
    max_date = records_df['TIMESTAMP'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        records_df = records_df[
            (records_df['TIMESTAMP'].dt.date >= start_date) & 
            (records_df['TIMESTAMP'].dt.date <= end_date)
        ]

# Model filter
if not apps_df.empty:
    available_models = ['All'] + list(apps_df['MODEL'].unique())
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        available_models,
        default=available_models[:3] if len(available_models) > 2 else available_models
    )
    
    if 'All' not in selected_models and selected_models:
        filtered_apps = apps_df[apps_df['MODEL'].isin(selected_models)]
        if not analysis_df.empty:
            analysis_df = analysis_df[analysis_df['MODEL'].isin(selected_models)]

# Feedback type filter
if not feedbacks_df.empty:
    feedback_types = ['All'] + list(feedbacks_df['NAME'].unique())
    selected_feedback_types = st.sidebar.multiselect(
        "Select Feedback Types",
        feedback_types,
        default=['All']
    )

# Enhanced key metrics with Power BI style
st.markdown("### 📊 Key Performance Indicators")

# Extract comprehensive metrics from JSON data
def extract_comprehensive_metrics(records_df, feedbacks_df, analysis_df):
    """Extract detailed metrics from JSON fields"""
    metrics = {
        'total_questions': len(records_df) if not records_df.empty else 0,
        'avg_latency': 0,
        'total_cost': 0,
        'avg_cost': 0,
        'total_tokens': 0,
        'avg_tokens': 0,
        'avg_feedback_score': 0,
        'success_rate': 0,
        'total_requests': 0,
        'completion_tokens': 0,
        'prompt_tokens': 0
    }
    
    if not records_df.empty:
        metrics['avg_latency'] = records_df['LATENCY'].mean()
        metrics['total_cost'] = records_df['COST'].sum()
        metrics['avg_cost'] = records_df['COST'].mean()
        metrics['total_tokens'] = records_df['TOKENS'].sum()
        metrics['avg_tokens'] = records_df['TOKENS'].mean()
        
        # Extract detailed cost/token info from JSON
        for _, row in records_df.iterrows():
            cost_json = safe_json_parse(row.get('COST_JSON', '{}'))
            perf_json = safe_json_parse(row.get('PERF_JSON', '{}'))
            
            metrics['total_requests'] += cost_json.get('n_requests', 0)
            metrics['completion_tokens'] += cost_json.get('n_completion_tokens', 0)
            metrics['prompt_tokens'] += cost_json.get('n_prompt_tokens', 0)
    
    if not analysis_df.empty:
        metrics['avg_feedback_score'] = analysis_df['RESULT'].mean()
        
    if not feedbacks_df.empty:
        completed = len(feedbacks_df[feedbacks_df['STATUS'] == 'done'])
        metrics['success_rate'] = (completed / len(feedbacks_df)) * 100 if len(feedbacks_df) > 0 else 0
    
    return metrics

metrics = extract_comprehensive_metrics(records_df, feedbacks_df, analysis_df)

# Create Power BI style metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['total_questions']:,}</div>
        <div class="metric-label">Total Questions</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['avg_feedback_score']:.2f}</div>
        <div class="metric-label">Avg Score</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['avg_latency']:.1f}s</div>
        <div class="metric-label">Avg Latency</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['total_cost']:.4f}</div>
        <div class="metric-label">Total Cost</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['success_rate']:.1f}%</div>
        <div class="metric-label">Success Rate</div>
    </div>
    """, unsafe_allow_html=True)

# Additional metrics row
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['total_tokens']:,}</div>
        <div class="metric-label">Total Tokens</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['total_requests']:,}</div>
        <div class="metric-label">API Requests</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['prompt_tokens']:,}</div>
        <div class="metric-label">Prompt Tokens</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{metrics['completion_tokens']:,}</div>
        <div class="metric-label">Completion Tokens</div>
    </div>
    """, unsafe_allow_html=True)

# Model Performance Comparison
st.markdown('<div class="model-comparison-header">🔄 Model Performance Comparison</div>', 
           unsafe_allow_html=True)

if not analysis_df.empty and len(analysis_df['MODEL'].unique()) > 1:
    # Model performance comparison
    model_metrics = analysis_df.groupby('MODEL').agg({
        'RESULT': ['mean', 'std', 'count'],
        'LATENCY': 'mean',
        'COST': 'mean',
        'TOKENS': 'mean'
    }).round(3)
    
    model_metrics.columns = ['Avg_Score', 'Score_Std', 'Total_Evaluations', 
                           'Avg_Latency', 'Avg_Cost', 'Avg_Tokens']
    model_metrics = model_metrics.reset_index()
    
    # Display comparison table
    st.subheader("📊 Model Performance Summary")
    st.dataframe(model_metrics, use_container_width=True)
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_score = px.bar(
            model_metrics, 
            x='MODEL', 
            y='Avg_Score',
            title="Average Feedback Score by Model",
            color='Avg_Score',
            color_continuous_scale='Viridis'
        )
        fig_score.update_layout(height=400)
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        fig_latency = px.bar(
            model_metrics, 
            x='MODEL', 
            y='Avg_Latency',
            title="Average Latency by Model",
            color='Avg_Latency',
            color_continuous_scale='Reds'
        )
        fig_latency.update_layout(height=400)
        st.plotly_chart(fig_latency, use_container_width=True)

# Feedback Distribution Analysis
st.markdown('<div class="section-header">📈 Feedback Analysis</div>', unsafe_allow_html=True)

if not analysis_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Feedback score distribution
        fig_dist = px.histogram(
            analysis_df, 
            x='RESULT', 
            color='MODEL',
            title="Feedback Score Distribution by Model",
            nbins=20,
            barmode='overlay'
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Box plot of scores by feedback type
        fig_box = px.box(
            analysis_df, 
            x='NAME', 
            y='RESULT',
            color='MODEL',
            title="Score Distribution by Feedback Type"
        )
        fig_box.update_xaxes(tickangle=45)
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

# Performance over time
if not analysis_df.empty:
    st.subheader("📅 Performance Trends Over Time")
    
    # Daily aggregation
    daily_performance = analysis_df.groupby([
        analysis_df['TIMESTAMP'].dt.date, 'MODEL'
    ]).agg({
        'RESULT': 'mean',
        'LATENCY': 'mean'
    }).reset_index()
    daily_performance.columns = ['Date', 'Model', 'Avg_Score', 'Avg_Latency']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_trend_score = px.line(
            daily_performance,
            x='Date',
            y='Avg_Score',
            color='Model',
            title="Average Score Trends",
            markers=True
        )
        st.plotly_chart(fig_trend_score, use_container_width=True)
    
    with col2:
        fig_trend_latency = px.line(
            daily_performance,
            x='Date',
            y='Avg_Latency',
            color='Model',
            title="Latency Trends",
            markers=True
        )
        st.plotly_chart(fig_trend_latency, use_container_width=True)

# Complexity Analysis
st.markdown('<div class="section-header">🎯 Complexity Analysis</div>', unsafe_allow_html=True)

if not analysis_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by complexity
        complexity_performance = analysis_df.groupby(['COMPLEXITY', 'MODEL'])['RESULT'].mean().reset_index()
        fig_complexity = px.bar(
            complexity_performance,
            x='COMPLEXITY',
            y='RESULT',
            color='MODEL',
            title="Performance by Question Complexity",
            barmode='group'
        )
        st.plotly_chart(fig_complexity, use_container_width=True)
    
    with col2:
        # Latency by complexity
        if 'LATENCY' in analysis_df.columns:
            complexity_latency = analysis_df.groupby(['COMPLEXITY', 'MODEL'])['LATENCY'].mean().reset_index()
            fig_complexity_latency = px.bar(
                complexity_latency,
                x='COMPLEXITY',
                y='LATENCY',
                color='MODEL',
                title="Latency by Question Complexity",
                barmode='group'
            )
            st.plotly_chart(fig_complexity_latency, use_container_width=True)

# Comprehensive Cost & Token Analysis
st.markdown('<div class="section-header">💰 Cost & Token Analysis</div>', unsafe_allow_html=True)

if not records_df.empty:
    # Use the analysis_df which already has model information merged
    if not analysis_df.empty:
        # Cost and Token Visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Total cost by model
            cost_by_model = analysis_df.groupby('MODEL')['COST'].sum().reset_index()
            cost_by_model = cost_by_model[cost_by_model['COST'] > 0]  # Only show models with cost data
            
            if not cost_by_model.empty:
                fig_cost_pie = px.pie(
                    cost_by_model,
                    values='COST',
                    names='MODEL',
                    title='Total Cost Distribution by Model',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_cost_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_cost_pie, use_container_width=True)
            else:
                st.info("No cost data available for visualization")
        
        with col2:
            # Token usage by model
            token_by_model = analysis_df.groupby('MODEL')['TOKENS'].sum().reset_index()
            token_by_model = token_by_model[token_by_model['TOKENS'] > 0]  # Only show models with token data
            
            if not token_by_model.empty:
                fig_token_pie = px.pie(
                    token_by_model,
                    values='TOKENS',
                    names='MODEL',
                    title='Token Usage Distribution by Model',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_token_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_token_pie, use_container_width=True)
            else:
                st.info("No token data available for visualization")
        
        with col3:
            # Cost per token efficiency
            efficiency_data = analysis_df.groupby('MODEL').agg({
                'COST': 'sum',
                'TOKENS': 'sum'
            }).reset_index()
            efficiency_data = efficiency_data[(efficiency_data['TOKENS'] > 0) & (efficiency_data['COST'] > 0)]
            
            if not efficiency_data.empty:
                efficiency_data['cost_per_token'] = efficiency_data['COST'] / efficiency_data['TOKENS']
                fig_efficiency = px.bar(
                    efficiency_data,
                    x='MODEL',
                    y='cost_per_token',
                    title='Cost Efficiency (Cost per Token)',
                    color='cost_per_token',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
            else:
                st.info("Insufficient data for cost efficiency calculation")
        
        # Detailed cost breakdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost over time
            if 'TIMESTAMP' in analysis_df.columns and analysis_df['COST'].sum() > 0:
                daily_cost = analysis_df.groupby([
                    analysis_df['TIMESTAMP'].dt.date, 'MODEL'
                ])['COST'].sum().reset_index()
                daily_cost.columns = ['Date', 'Model', 'Daily_Cost']
                daily_cost = daily_cost[daily_cost['Daily_Cost'] > 0]
                
                if not daily_cost.empty:
                    fig_cost_trend = px.line(
                        daily_cost,
                        x='Date',
                        y='Daily_Cost',
                        color='Model',
                        title='Daily Cost Trends by Model',
                        markers=True
                    )
                    fig_cost_trend.update_layout(height=400)
                    st.plotly_chart(fig_cost_trend, use_container_width=True)
                else:
                    st.info("No cost trend data available")
            else:
                st.info("No timestamp or cost data available for trends")
        
        with col2:
            # Model comparison by cost and performance
            if len(analysis_df['MODEL'].unique()) > 1:
                model_comparison = analysis_df.groupby('MODEL').agg({
                    'COST': 'mean',
                    'RESULT': 'mean',
                    'LATENCY': 'mean',
                    'TOKENS': 'mean'
                }).reset_index()
                
                fig_comparison = px.scatter(
                    model_comparison,
                    x='COST',
                    y='RESULT',
                    size='TOKENS',
                    color='LATENCY',
                    hover_name='MODEL',
                    title='Model Performance vs Cost',
                    labels={'COST': 'Average Cost', 'RESULT': 'Average Score'},
                    color_continuous_scale='Viridis'
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.info("Need multiple models for comparison")
        
        # Comprehensive summary table
        st.subheader("📊 Cost & Token Summary by Model")
        
        cost_token_summary = analysis_df.groupby('MODEL').agg({
            'COST': ['sum', 'mean', 'std'],
            'TOKENS': ['sum', 'mean'],
            'RESULT': 'mean',
            'LATENCY': 'mean'
        }).round(4)
        
        # Flatten column names
        cost_token_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cost_token_summary.columns]
        cost_token_summary = cost_token_summary.reset_index()
        
        # Add efficiency metrics where possible
        cost_token_summary['cost_per_token'] = cost_token_summary.apply(
            lambda x: x['COST_sum'] / x['TOKENS_sum'] if x['TOKENS_sum'] > 0 else 0, axis=1
        )
        
        # Display with better column names
        display_summary = cost_token_summary.rename(columns={
            'MODEL': 'Model',
            'COST_sum': 'Total Cost',
            'COST_mean': 'Avg Cost/Query',
            'COST_std': 'Cost Std Dev',
            'TOKENS_sum': 'Total Tokens',
            'TOKENS_mean': 'Avg Tokens/Query',
            'RESULT_mean': 'Avg Performance Score',
            'LATENCY_mean': 'Avg Latency (s)',
            'cost_per_token': 'Cost per Token'
        })
        
        st.dataframe(display_summary, use_container_width=True)
    
    else:
        st.warning("No analysis data available. Please check if feedbacks and records data are properly merged.")
        
        # Show what data we do have
        if not records_df.empty:
            st.write("**Available Records Data Sample:**")
            st.dataframe(records_df[['RECORD_ID', 'APP_ID', 'COST', 'TOKENS', 'LATENCY']].head())
        
        if not apps_df.empty:
            st.write("**Available Apps Data:**")
            st.dataframe(apps_df[['APP_ID', 'APP_NAME', 'MODEL']])

# Advanced Record Analysis using JSON data
st.markdown('<div class="section-header">🔍 Advanced Record Analysis</div>', unsafe_allow_html=True)

if not records_df.empty:
    # Extract comprehensive information from RECORD_JSON
    detailed_records = []
    
    for _, row in records_df.iterrows():
        record_json = safe_json_parse(row.get('RECORD_JSON', '{}'))
        cost_json = safe_json_parse(row.get('COST_JSON', '{}'))
        perf_json = safe_json_parse(row.get('PERF_JSON', '{}'))
        
        # Extract metadata
        meta = record_json.get('meta', {})
        cost_info = cost_json if cost_json else record_json.get('cost', {})
        perf_info = perf_json if perf_json else record_json.get('perf', {})
        
        detailed_record = {
            'RECORD_ID': row['RECORD_ID'],
            'APP_ID': row['APP_ID'],
            'INPUT': row['INPUT'][:100] + "..." if len(row['INPUT']) > 100 else row['INPUT'],
            'OUTPUT': row['OUTPUT'][:100] + "..." if len(row['OUTPUT']) > 100 else row['OUTPUT'],
            'TIMESTAMP': row['TIMESTAMP'],
            'LATENCY': row['LATENCY'],
            
            # From metadata
            'semantic_model_file': meta.get('Semantic_Model_File', 'N/A'),
            'summarization_llm': meta.get('Summarization_LLM', 'N/A'),
            'user_input': meta.get('user_input', 'N/A'),
            
            # Cost information
            'cost': cost_info.get('cost', 0),
            'tokens': cost_info.get('n_tokens', 0),
            'requests': cost_info.get('n_requests', 0),
            'successful_requests': cost_info.get('n_successful_requests', 0),
            'prompt_tokens': cost_info.get('n_prompt_tokens', 0),
            'completion_tokens': cost_info.get('n_completion_tokens', 0),
            
            # Performance information
            'start_time': perf_info.get('start_time', 'N/A'),
            'end_time': perf_info.get('end_time', 'N/A')
        }
        
        detailed_records.append(detailed_record)
    
    detailed_records_df = pd.DataFrame(detailed_records)
    
    # Show detailed records analysis
    if not detailed_records_df.empty:
        st.subheader("📋 Detailed Records with JSON Analysis")
        
        # Add filtering options
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by LLM
            llms = detailed_records_df['summarization_llm'].unique()
            selected_llm = st.selectbox("Filter by LLM", ['All'] + list(llms))
            
        with col2:
            # Filter by cost range
            if detailed_records_df['cost'].max() > 0:
                cost_range = st.slider(
                    "Cost Range",
                    float(detailed_records_df['cost'].min()),
                    float(detailed_records_df['cost'].max()),
                    (float(detailed_records_df['cost'].min()), float(detailed_records_df['cost'].max())),
                    format="%.6f"
                )
        
        # Apply filters
        filtered_detailed_df = detailed_records_df.copy()
        
        if selected_llm != 'All':
            filtered_detailed_df = filtered_detailed_df[filtered_detailed_df['summarization_llm'] == selected_llm]
        
        if 'cost_range' in locals() and detailed_records_df['cost'].max() > 0:
            filtered_detailed_df = filtered_detailed_df[
                (filtered_detailed_df['cost'] >= cost_range[0]) & 
                (filtered_detailed_df['cost'] <= cost_range[1])
            ]
        
        # Display filtered results
        st.dataframe(filtered_detailed_df, use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Records Shown", len(filtered_detailed_df))
            
        with col2:
            if len(filtered_detailed_df) > 0:
                st.metric("Avg Cost", f"{filtered_detailed_df['cost'].mean():.6f}")
                
        with col3:
            if len(filtered_detailed_df) > 0:
                st.metric("Avg Latency", f"{filtered_detailed_df['LATENCY'].mean():.2f}s")

# Feedback reasoning analysis
if not analysis_df.empty and 'REASONING' in analysis_df.columns:
    st.subheader("💭 Feedback Reasoning Analysis")
    
    # Show sample reasoning for each model
    for model in analysis_df['MODEL'].unique():
        if pd.notna(model):
            with st.expander(f"Sample Reasoning - {model}"):
                model_data = analysis_df[analysis_df['MODEL'] == model]
                sample_reasoning = model_data[model_data['REASONING'].str.len() > 10]['REASONING'].head(3)
                for i, reasoning in enumerate(sample_reasoning):
                    st.write(f"**Example {i+1}:**")
                    st.write(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
                    st.write("---")

# Raw data exploration
with st.expander("🗂️ Explore Raw Data"):
    st.subheader("Data Tables")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Records", "Feedbacks", "Apps", "Feedback Definitions"])
    
    with tab1:
        if not records_df.empty:
            st.dataframe(records_df, use_container_width=True)
    
    with tab2:
        if not feedbacks_df.empty:
            st.dataframe(feedbacks_df, use_container_width=True)
    
    with tab3:
        if not apps_df.empty:
            st.dataframe(apps_df, use_container_width=True)
    
    with tab4:
        if not feedback_defs_df.empty:
            st.dataframe(feedback_defs_df, use_container_width=True)

# Export functionality
st.markdown('<div class="section-header">📤 Export Data</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

if not analysis_df.empty:
    with col1:
        # Model comparison export
        if st.button("📊 Export Model Comparison"):
            if 'model_metrics' in locals():
                csv = model_metrics.to_csv(index=False)
                st.download_button(
                    label="Download Model Comparison CSV",
                    data=csv,
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        # Full analysis export
        if st.button("📋 Export Full Analysis"):
            csv = analysis_df.to_csv(index=False)
            st.download_button(
                label="Download Full Analysis CSV",
                data=csv,
                file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Export to Snowflake table
        if st.button("❄️ Save Analysis to Snowflake"):
            try:
                session = create_snowpark_session()
                if session:
                    # Create analysis summary table
                    analysis_summary = analysis_df.groupby('MODEL').agg({
                        'RESULT': ['mean', 'std', 'count'],
                        'LATENCY': 'mean',
                        'COST': 'mean'
                    }).round(4)
                    
                    analysis_summary.columns = ['avg_score', 'score_std', 'total_queries', 'avg_latency', 'avg_cost']
                    analysis_summary = analysis_summary.reset_index()
                    analysis_summary['analysis_date'] = datetime.now()
                    
                    # Write to Snowflake
                    table_name = f"{DATABASE}.{SCHEMA}.TRULENS_ANALYSIS_SUMMARY"
                    session.write_pandas(
                        analysis_summary, 
                        table_name,
                        auto_create_table=True,
                        overwrite=True
                    )
                    st.success(f"✅ Analysis saved to {table_name}!")
                else:
                    st.error("❌ Cannot save to Snowflake - connection failed")
            except Exception as e:
                st.error(f"❌ Error saving to Snowflake: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: {DATABASE}.{SCHEMA}*")