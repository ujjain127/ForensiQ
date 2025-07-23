"""
System Analytics Page - ForensiQ Officer Interface
=================================================
Advanced system analytics and performance monitoring for officers
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

def show_system_analytics_page():
    st.markdown("""
    <div class="main-header">
        <h1 class="app-title">📊 System Analytics</h1>
        <p style="font-size: 1.1rem; opacity: 0.9; margin: 0;">Platform Performance & Usage Statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get complaints data
    all_complaints = st.session_state.complaints_db
    
    if not all_complaints:
        st.markdown("""
        <div class="complaint-card" style="text-align: center;">
            <h3>📈 No Analytics Data Available</h3>
            <p>System analytics will be available once there are active cases in the system.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["🔍 Case Analytics", "⚡ Performance", "👥 User Activity"])
    
    with tab1:
        show_case_analytics(all_complaints)
    
    with tab2:
        show_performance_analytics()
    
    with tab3:
        show_user_analytics()

def show_case_analytics(complaints):
    """Show case-related analytics"""
    
    # Calculate metrics
    total_cases = len(complaints)
    df = pd.DataFrame(complaints)
    
    # Daily case volume chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #8b5cf6;">📈 Daily Case Volume</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create daily volume data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        daily_volumes = np.random.poisson(lam=total_cases/30, size=len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_volumes,
            mode='lines+markers',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=6),
            name='Daily Cases'
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #8b5cf6;">🎯 Case Resolution Rate</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Resolution rate pie chart
        status_counts = df['status'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker_colors=['#8b5cf6', '#06b6d4', '#10b981', '#f97316']
        )])
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Priority distribution
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #8b5cf6;">⚠️ Priority Distribution Analysis</h4>
    </div>
    """, unsafe_allow_html=True)
    
    priority_data = df['priority_score'].value_counts().sort_index()
    
    fig = go.Figure(data=[go.Bar(
        x=priority_data.index,
        y=priority_data.values,
        marker_color=['#10b981', '#f59e0b', '#f97316', '#ef4444']
    )])
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        height=400,
        xaxis_title="Priority Score",
        yaxis_title="Number of Cases"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_performance_analytics():
    """Show system performance analytics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card" style="background: linear-gradient(135deg, #10b981 0%, #047857 100%);">
            <h3 class="stats-number">99.8%</h3>
            <p class="stats-label">System Uptime</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);">
            <h3 class="stats-number">1.2s</h3>
            <p class="stats-label">Avg Response Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
            <h3 class="stats-number">156</h3>
            <p class="stats-label">Active Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card" style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);">
            <h3 class="stats-number">2.1GB</h3>
            <p class="stats-label">Storage Used</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #8b5cf6;">📊 Performance Trends (Last 24 Hours)</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate mock performance data
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    response_times = np.random.normal(1.2, 0.3, len(hours))
    cpu_usage = np.random.normal(45, 10, len(hours))
    memory_usage = np.random.normal(60, 15, len(hours))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=response_times,
        mode='lines', name='Response Time (s)',
        line=dict(color='#8b5cf6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=cpu_usage,
        mode='lines', name='CPU Usage (%)',
        line=dict(color='#06b6d4', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        height=400,
        yaxis=dict(title='Response Time (seconds)', side='left'),
        yaxis2=dict(title='CPU Usage (%)', side='right', overlaying='y'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_user_analytics():
    """Show user activity analytics"""
    
    # Mock user data
    user_data = {
        'User Type': ['Public Users', 'Security Officers', 'Administrators'],
        'Active': [142, 12, 2],
        'Total': [1200, 25, 5]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #8b5cf6;">👥 User Distribution</h4>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Bar(
            x=user_data['User Type'],
            y=user_data['Active'],
            marker_color=['#10b981', '#f59e0b', '#ef4444'],
            name='Active Users'
        )])
        
        fig.add_trace(go.Bar(
            x=user_data['User Type'],
            y=[total - active for total, active in zip(user_data['Total'], user_data['Active'])],
            marker_color=['rgba(16, 185, 129, 0.3)', 'rgba(245, 158, 11, 0.3)', 'rgba(239, 68, 68, 0.3)'],
            name='Inactive Users'
        ))
        
        fig.update_layout(
            barmode='stack',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #8b5cf6;">⏰ Peak Usage Hours</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate hourly usage data
        hours = list(range(24))
        usage = [20, 15, 10, 8, 12, 25, 45, 65, 85, 90, 88, 92, 95, 88, 85, 82, 78, 75, 65, 55, 45, 35, 30, 25]
        
        fig = go.Figure(data=[go.Scatter(
            x=hours,
            y=usage,
            mode='lines+markers',
            fill='tonexty',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=6)
        )])
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            height=350,
            xaxis_title="Hour of Day",
            yaxis_title="Active Users"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity log
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #8b5cf6;">📋 Recent System Activity</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Mock activity data
    activities = [
        {"Time": "2 minutes ago", "Event": "New case submitted", "User": "john_doe", "Type": "Case"},
        {"Time": "5 minutes ago", "Event": "Analysis completed", "User": "officer_smith", "Type": "Analysis"},
        {"Time": "12 minutes ago", "Event": "User login", "User": "jane_smith", "Type": "Auth"},
        {"Time": "18 minutes ago", "Event": "Priority updated", "User": "officer_smith", "Type": "Case"},
        {"Time": "25 minutes ago", "Event": "Evidence uploaded", "User": "alex_user", "Type": "Evidence"},
    ]
    
    for activity in activities:
        st.markdown(f"""
        <div class="timeline-event">
            <strong>{activity['Event']}</strong><br>
            <small style="color: #94a3b8;">👤 {activity['User']} • 🏷️ {activity['Type']} • ⏰ {activity['Time']}</small>
        </div>
        """, unsafe_allow_html=True)
