"""
Case Overview Page - ForensiQ Officer Interface  
==============================================
Provides comprehensive overview of all cases for cyber security officers
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

def show_case_overview_page():
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1 class="app-title">📊 Case Overview Dashboard</h1>
        <p style="font-size: 1.1rem; opacity: 0.9; margin: 0;">Real-time Intelligence & Case Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all complaints
    all_complaints = st.session_state.complaints_db
    
    if not all_complaints:
        st.markdown("""
        <div class="complaint-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #94a3b8; margin-bottom: 1rem;">📋 No Active Cases</h3>
            <p style="color: #64748b;">Cases will appear here once users submit complaints through the system.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Professional dashboard metrics
    show_dashboard_metrics(all_complaints)
    
    # Charts and visualizations
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_case_status_chart(all_complaints)
        show_priority_distribution(all_complaints)
    
    with col2:
        show_case_timeline(all_complaints)
        show_incident_types_chart(all_complaints)
    
    # Professional case management table
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    show_case_management_table(all_complaints)

def show_dashboard_metrics(complaints):
    """Display key metrics in professional cards like Fortexa dashboard"""
    
    # Calculate metrics
    total_cases = len(complaints)
    new_cases = len([c for c in complaints if c['status'] == 'submitted'])
    under_review = len([c for c in complaints if c['status'] == 'under_review'])
    completed_cases = len([c for c in complaints if c['status'] == 'completed'])
    high_priority = len([c for c in complaints if c['priority_score'] >= 7])
    
    # Cases in last 24 hours
    now = datetime.now()
    recent_cases = len([
        c for c in complaints 
        if datetime.fromisoformat(c['timestamp']) > now - timedelta(hours=24)
    ])
    
    # Calculate threat percentage and risk score
    threat_percentage = min(100, (high_priority / total_cases * 100) + 25) if total_cases > 0 else 0
    risk_score = min(100, (high_priority * 10) + (under_review * 5) + 30)
    
    # Create professional metric cards grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
            <h3 class="stats-number">{threat_percentage:.0f}%</h3>
            <p class="stats-label">Total Threats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate weekly change
        weekly_change = 17  # Mock positive change
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);">
            <h3 class="stats-number">{weekly_change}%</h3>
            <p class="stats-label">Video file risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        image_risk = min(100, (new_cases / total_cases * 100) + 20) if total_cases > 0 else 0
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);">
            <h3 class="stats-number">{image_risk:.0f}%</h3>
            <p class="stats-label">Image file risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        docs_risk = min(100, (under_review / total_cases * 100) + 15) if total_cases > 0 else 0
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);">
            <h3 class="stats-number">{docs_risk:.0f}%</h3>
            <p class="stats-label">Docs file risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Score gauge (similar to Fortexa)
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    col_gauge, col_metrics = st.columns([1, 2])
    
    with col_gauge:
        # Create risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'color': 'white', 'size': 20}},
            delta = {'reference': 50, 'increasing': {'color': "#ff4444"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                'bar': {'color': "#8b5cf6"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#475569",
                'steps': [
                    {'range': [0, 50], 'color': "#1e293b"},
                    {'range': [50, 80], 'color': "#334155"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}
            }))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=300
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_metrics:
        # Additional metrics cards
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #8b5cf6; margin-bottom: 1rem;">📈 Case Statistics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Total Cases</p>
                    <p style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">{total_cases}</p>
                </div>
                <div>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">High Priority</p>
                    <p style="color: #ff4444; margin: 0; font-size: 1.5rem; font-weight: 700;">{high_priority}</p>
                </div>
                <div>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Under Review</p>
                    <p style="color: #60a5fa; margin: 0; font-size: 1.5rem; font-weight: 700;">{under_review}</p>
                </div>
                <div>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Completed</p>
                    <p style="color: #34d399; margin: 0; font-size: 1.5rem; font-weight: 700;">{completed_cases}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_case_status_chart(complaints):
    """Display professional case status distribution chart"""
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #8b5cf6; margin-bottom: 1rem;">� Case Status Distribution</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Count status
    status_counts = Counter(c['status'] for c in complaints)
    status_labels = {
        'submitted': 'New Cases',
        'under_review': 'Under Review', 
        'completed': 'Completed',
        'closed': 'Closed'
    }
    
    labels = [status_labels.get(status, status.title()) for status in status_counts.keys()]
    values = list(status_counts.values())
    colors = ['#8b5cf6', '#06b6d4', '#10b981', '#f97316']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont={'color': 'white', 'size': 12}
    )])
    
    fig.update_layout(
        showlegend=True,
        legend={'font': {'color': 'white'}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        height=350
    )
    
    fig.update_layout(
        showlegend=True,
        height=300,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_priority_distribution(complaints):
    """Display priority score distribution"""
    st.markdown("### ⚡ Priority Distribution")
    
    priority_scores = [c['priority_score'] for c in complaints]
    
    # Create priority bins
    priority_bins = {'Low (1-3)': 0, 'Medium (4-6)': 0, 'High (7-8)': 0, 'Critical (9-10)': 0}
    
    for score in priority_scores:
        if score <= 3:
            priority_bins['Low (1-3)'] += 1
        elif score <= 6:
            priority_bins['Medium (4-6)'] += 1
        elif score <= 8:
            priority_bins['High (7-8)'] += 1
        else:
            priority_bins['Critical (9-10)'] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(priority_bins.keys()),
            y=list(priority_bins.values()),
            marker_color=['#4CAF50', '#FFC107', '#FF9800', '#F44336']
        )
    ])
    
    fig.update_layout(
        xaxis_title="Priority Level",
        yaxis_title="Number of Cases", 
        height=300,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_case_timeline(complaints):
    """Display case submission timeline"""
    st.markdown("### 📅 Case Submission Timeline")
    
    # Group by date
    dates = [datetime.fromisoformat(c['timestamp']).date() for c in complaints]
    date_counts = Counter(dates)
    
    # Fill in missing dates for last 30 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=29)
    
    date_range = []
    counts = []
    current_date = start_date
    
    while current_date <= end_date:
        date_range.append(current_date)
        counts.append(date_counts.get(current_date, 0))
        current_date += timedelta(days=1)
    
    fig = go.Figure(data=[
        go.Scatter(
            x=date_range,
            y=counts,
            mode='lines+markers',
            line=dict(color='#1e3c72', width=2),
            marker=dict(size=6)
        )
    ])
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cases Submitted",
        height=300,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_incident_types_chart(complaints):
    """Display incident type distribution"""
    st.markdown("### 🔐 Incident Types")
    
    incident_types = [c['incident']['type'] for c in complaints]
    type_counts = Counter(incident_types)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(type_counts.values()),
            y=list(type_counts.keys()),
            orientation='h',
            marker_color='#2196F3'
        )
    ])
    
    fig.update_layout(
        xaxis_title="Number of Cases",
        yaxis_title="Incident Type",
        height=300,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_case_management_table(complaints):
    """Display interactive case management table"""
    st.markdown("### 📋 Case Management")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "New Cases", "Under Review", "Completed", "Closed"],
            key="case_overview_status"
        )
    
    with col2:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "Critical (9-10)", "High (7-8)", "Medium (4-6)", "Low (1-3)"],
            key="case_overview_priority"
        )
    
    with col3:
        assigned_filter = st.selectbox(
            "Filter by Assignment",
            ["All", "Assigned", "Unassigned"],
            key="case_overview_assigned"
        )
    
    with col4:
        search_term = st.text_input("Search cases", placeholder="Case ID, title, type...", key="case_overview_search")
    
    # Apply filters
    filtered_complaints = complaints.copy()
    
    # Status filter
    if status_filter != "All":
        status_map = {
            "New Cases": "submitted",
            "Under Review": "under_review",
            "Completed": "completed", 
            "Closed": "closed"
        }
        filtered_complaints = [c for c in filtered_complaints if c['status'] == status_map[status_filter]]
    
    # Priority filter
    if priority_filter != "All":
        priority_ranges = {
            "Critical (9-10)": (9, 10),
            "High (7-8)": (7, 8),
            "Medium (4-6)": (4, 6),
            "Low (1-3)": (1, 3)
        }
        min_p, max_p = priority_ranges[priority_filter]
        filtered_complaints = [c for c in filtered_complaints if min_p <= c['priority_score'] <= max_p]
    
    # Assignment filter
    if assigned_filter == "Assigned":
        filtered_complaints = [c for c in filtered_complaints if c.get('assigned_officer')]
    elif assigned_filter == "Unassigned":
        filtered_complaints = [c for c in filtered_complaints if not c.get('assigned_officer')]
    
    # Search filter
    if search_term:
        filtered_complaints = [
            c for c in filtered_complaints
            if search_term.lower() in c['complaint_id'].lower()
            or search_term.lower() in c['incident']['title'].lower()
            or search_term.lower() in c['incident']['type'].lower()
        ]
    
    # Create table data
    table_data = []
    for complaint in filtered_complaints:
        
        # Status styling
        status_icons = {
            'submitted': '🆕',
            'under_review': '🔄',
            'completed': '✅',
            'closed': '🔒'
        }
        
        # Priority styling
        priority = complaint['priority_score']
        if priority >= 9:
            priority_display = f"🔴 {priority}"
        elif priority >= 7:
            priority_display = f"🟠 {priority}"
        elif priority >= 4:
            priority_display = f"🟡 {priority}"
        else:
            priority_display = f"🟢 {priority}"
        
        table_data.append({
            'Case ID': complaint['complaint_id'],
            'Title': complaint['incident']['title'][:50] + ('...' if len(complaint['incident']['title']) > 50 else ''),
            'Type': complaint['incident']['type'],
            'Status': f"{status_icons.get(complaint['status'], '❓')} {complaint['status'].title()}",
            'Priority': priority_display,
            'Submitted': datetime.fromisoformat(complaint['timestamp']).strftime('%Y-%m-%d'),
            'Assigned To': complaint.get('assigned_officer', 'Unassigned'),
            'Evidence': len(complaint['evidence_files'])
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Display table with selection
        selected_indices = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        # Action buttons for selected case
        if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
            selected_idx = selected_indices.selection.rows[0]
            selected_complaint = filtered_complaints[selected_idx]
            
            st.markdown("### 🔧 Case Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔍 View Details", key="view_case_details"):
                    st.session_state.selected_case = selected_complaint
                    st.session_state.current_page = 'case_details'
                    st.rerun()
            
            with col2:
                if st.button("🤖 Auto Analysis", key="start_auto_analysis"):
                    st.session_state.selected_case = selected_complaint
                    st.session_state.current_page = 'auto_analysis'
                    st.rerun()
            
            with col3:
                if st.button("📊 Timeline", key="view_timeline"):
                    st.session_state.selected_case = selected_complaint
                    st.session_state.current_page = 'timeline_reconstruction'
                    st.rerun()
            
            with col4:
                if st.button("🕸️ Graph", key="view_graph"):
                    st.session_state.selected_case = selected_complaint
                    st.session_state.current_page = 'actor_graph'
                    st.rerun()
    
    else:
        st.info("No cases match the current filters.")
    
    # Bulk actions
    if table_data:
        st.markdown("### ⚡ Bulk Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh All Data"):
                st.rerun()
        
        with col2:
            if st.button("📤 Export Cases"):
                st.info("Export functionality will be implemented in the next update.")
        
        with col3:
            if st.button("📈 Generate Report"):
                st.info("Report generation will be implemented in the next update.")
