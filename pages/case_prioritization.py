"""
Case Prioritization Page - ForensiQ Officer Interface
====================================================
Advanced case prioritization using ML algorithms and risk assessment
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import Counter

def show_case_prioritization_page():
    st.markdown('<div class="main-header"><h2>⚡ Case Prioritization</h2><p>AI-powered case prioritization and resource allocation</p></div>', unsafe_allow_html=True)
    
    # Get all complaints
    all_complaints = st.session_state.complaints_db
    
    if not all_complaints:
        st.info("No cases available for prioritization. Cases will appear here once users submit complaints.")
        return
    
    # Calculate enhanced priority scores
    enhanced_complaints = calculate_enhanced_priorities(all_complaints)
    
    # Priority dashboard
    show_priority_dashboard(enhanced_complaints)
    
    st.markdown("---")
    
    # Priority matrix and visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        show_priority_matrix(enhanced_complaints)
    
    with col2:
        show_resource_allocation(enhanced_complaints)
    
    st.markdown("---")
    
    # Prioritized case queue
    show_prioritized_queue(enhanced_complaints)

def calculate_enhanced_priorities(complaints):
    """Calculate enhanced priority scores using multiple factors"""
    
    enhanced_complaints = []
    
    for complaint in complaints:
        # Base priority score
        base_score = complaint['priority_score']
        
        # Time-based urgency factor
        submitted_time = datetime.fromisoformat(complaint['timestamp'])
        hours_since_submission = (datetime.now() - submitted_time).total_seconds() / 3600
        
        # Urgency multiplier based on time elapsed
        if hours_since_submission > 72:  # 3 days
            time_urgency = 1.5
        elif hours_since_submission > 24:  # 1 day
            time_urgency = 1.2
        else:
            time_urgency = 1.0
        
        # Evidence complexity factor
        evidence_count = len(complaint['evidence_files'])
        if evidence_count > 5:
            evidence_factor = 1.3
        elif evidence_count > 2:
            evidence_factor = 1.1
        else:
            evidence_factor = 1.0
        
        # Incident type risk factor
        incident_type = complaint['incident']['type'].lower()
        type_risk_factors = {
            'ransomware attack': 2.0,
            'data breach': 1.8,
            'financial fraud': 1.6,
            'identity theft': 1.5,
            'phishing attack': 1.4,
            'malware infection': 1.3,
            'unauthorized access': 1.2,
            'social engineering': 1.1,
            'other': 1.0
        }
        
        type_factor = type_risk_factors.get(incident_type, 1.0)
        
        # Analysis status urgency
        analysis_status = complaint.get('analysis_status', 'pending')
        if analysis_status == 'pending' and hours_since_submission > 48:
            analysis_urgency = 1.3
        elif analysis_status == 'failed':
            analysis_urgency = 1.4
        else:
            analysis_urgency = 1.0
        
        # Calculate enhanced priority
        enhanced_priority = base_score * time_urgency * evidence_factor * type_factor * analysis_urgency
        enhanced_priority = min(10.0, enhanced_priority)  # Cap at 10
        
        # Assign priority category
        if enhanced_priority >= 8.5:
            priority_category = 'Critical'
            priority_color = '#F44336'
        elif enhanced_priority >= 7.0:
            priority_category = 'High'
            priority_color = '#FF9800'
        elif enhanced_priority >= 5.0:
            priority_category = 'Medium'
            priority_color = '#FFC107'
        else:
            priority_category = 'Low'
            priority_color = '#4CAF50'
        
        # Calculate estimated effort (mock calculation)
        estimated_effort_hours = calculate_estimated_effort(complaint, evidence_count)
        
        # Create enhanced complaint
        enhanced_complaint = complaint.copy()
        enhanced_complaint.update({
            'enhanced_priority': round(enhanced_priority, 2),
            'priority_category': priority_category,
            'priority_color': priority_color,
            'time_urgency': time_urgency,
            'evidence_factor': evidence_factor,
            'type_factor': type_factor,
            'analysis_urgency': analysis_urgency,
            'hours_since_submission': round(hours_since_submission, 1),
            'estimated_effort_hours': estimated_effort_hours
        })
        
        enhanced_complaints.append(enhanced_complaint)
    
    # Sort by enhanced priority
    enhanced_complaints.sort(key=lambda x: x['enhanced_priority'], reverse=True)
    
    return enhanced_complaints

def calculate_estimated_effort(complaint, evidence_count):
    """Calculate estimated investigation effort in hours"""
    
    base_hours = 4  # Base investigation time
    
    # Add time based on evidence complexity
    evidence_hours = evidence_count * 1.5
    
    # Add time based on incident type
    incident_type = complaint['incident']['type'].lower()
    type_hours = {
        'ransomware attack': 12,
        'data breach': 10,
        'financial fraud': 8,
        'identity theft': 6,
        'phishing attack': 4,
        'malware infection': 6,
        'unauthorized access': 5,
        'social engineering': 3,
        'other': 2
    }
    
    type_effort = type_hours.get(incident_type, 2)
    
    # Total estimated effort
    total_effort = base_hours + evidence_hours + type_effort
    
    return min(40, total_effort)  # Cap at 40 hours

def show_priority_dashboard(complaints):
    """Show priority dashboard metrics"""
    
    # Calculate metrics
    total_cases = len(complaints)
    critical_cases = len([c for c in complaints if c['priority_category'] == 'Critical'])
    high_cases = len([c for c in complaints if c['priority_category'] == 'High'])
    pending_analysis = len([c for c in complaints if c.get('analysis_status', 'pending') == 'pending'])
    
    # Average response time target vs actual
    avg_hours_pending = sum(c['hours_since_submission'] for c in complaints) / max(len(complaints), 1)
    
    # Total estimated workload
    total_workload = sum(c['estimated_effort_hours'] for c in complaints)
    
    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Cases", total_cases)
    
    with col2:
        st.metric("Critical", critical_cases, delta="🔴" if critical_cases > 0 else None)
    
    with col3:
        st.metric("High Priority", high_cases, delta="🟠" if high_cases > 0 else None)
    
    with col4:
        st.metric("Pending Analysis", pending_analysis)
    
    with col5:
        st.metric("Avg Age (hrs)", f"{avg_hours_pending:.1f}")
    
    with col6:
        st.metric("Total Workload", f"{total_workload:.0f}h")

def show_priority_matrix(complaints):
    """Show priority vs effort matrix"""
    st.markdown("### 📊 Priority-Effort Matrix")
    
    # Prepare data for scatter plot
    priorities = [c['enhanced_priority'] for c in complaints]
    efforts = [c['estimated_effort_hours'] for c in complaints]
    colors = [c['priority_color'] for c in complaints]
    case_ids = [c['complaint_id'] for c in complaints]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=efforts,
        y=priorities,
        mode='markers',
        marker=dict(
            size=12,
            color=colors,
            line=dict(width=1, color='white')
        ),
        text=case_ids,
        hovertemplate='<b>%{text}</b><br>Priority: %{y}<br>Effort: %{x}h<extra></extra>'
    ))
    
    # Add quadrant lines
    fig.add_hline(y=6.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=15, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=7, y=9, text="Quick Wins<br>(High Priority, Low Effort)", 
                      showarrow=False, bgcolor="rgba(76, 175, 80, 0.1)")
    fig.add_annotation(x=25, y=9, text="Major Projects<br>(High Priority, High Effort)", 
                      showarrow=False, bgcolor="rgba(244, 67, 54, 0.1)")
    fig.add_annotation(x=7, y=3, text="Fill-ins<br>(Low Priority, Low Effort)", 
                      showarrow=False, bgcolor="rgba(158, 158, 158, 0.1)")
    fig.add_annotation(x=25, y=3, text="Questionable<br>(Low Priority, High Effort)", 
                      showarrow=False, bgcolor="rgba(255, 152, 0, 0.1)")
    
    fig.update_layout(
        xaxis_title="Estimated Effort (hours)",
        yaxis_title="Enhanced Priority Score",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_resource_allocation(complaints):
    """Show resource allocation recommendations"""
    st.markdown("### 👥 Resource Allocation")
    
    # Calculate workload by priority category
    priority_workload = {}
    for complaint in complaints:
        category = complaint['priority_category']
        hours = complaint['estimated_effort_hours']
        priority_workload[category] = priority_workload.get(category, 0) + hours
    
    # Create pie chart
    labels = list(priority_workload.keys())
    values = list(priority_workload.values())
    colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50']  # Critical, High, Medium, Low
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.3,
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{percent}<br>%{value}h'
    )])
    
    fig.update_layout(
        title="Workload Distribution by Priority",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource recommendations
    st.markdown("#### 💡 Resource Recommendations")
    
    total_hours = sum(values)
    critical_hours = priority_workload.get('Critical', 0)
    high_hours = priority_workload.get('High', 0)
    
    if critical_hours > 0:
        st.error(f"🚨 {critical_hours}h of critical cases require immediate attention")
    
    if high_hours > 0:
        st.warning(f"⚠️ {high_hours}h of high-priority cases should be addressed today")
    
    if total_hours > 40:
        st.info(f"📋 Total workload ({total_hours}h) exceeds single analyst capacity - consider team assignment")

def show_prioritized_queue(complaints):
    """Show prioritized case queue with detailed information"""
    st.markdown("### 📋 Prioritized Case Queue")
    
    # Filter and sort options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "Critical", "High", "Medium", "Low"],
            key="priority_queue_filter"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Submitted", "Under Review", "Completed"],
            key="priority_status_filter"
        )
    
    with col3:
        sort_option = st.selectbox(
            "Sort by",
            ["Enhanced Priority", "Original Priority", "Submission Time", "Estimated Effort"],
            key="priority_sort"
        )
    
    # Apply filters
    filtered_complaints = complaints.copy()
    
    if priority_filter != "All":
        filtered_complaints = [c for c in filtered_complaints if c['priority_category'] == priority_filter]
    
    if status_filter != "All":
        status_map = {"Submitted": "submitted", "Under Review": "under_review", "Completed": "completed"}
        filtered_complaints = [c for c in filtered_complaints if c['status'] == status_map[status_filter]]
    
    # Apply sorting
    if sort_option == "Enhanced Priority":
        filtered_complaints.sort(key=lambda x: x['enhanced_priority'], reverse=True)
    elif sort_option == "Original Priority":
        filtered_complaints.sort(key=lambda x: x['priority_score'], reverse=True)
    elif sort_option == "Submission Time":
        filtered_complaints.sort(key=lambda x: x['timestamp'])
    elif sort_option == "Estimated Effort":
        filtered_complaints.sort(key=lambda x: x['estimated_effort_hours'])
    
    # Display cases
    if not filtered_complaints:
        st.info("No cases match the current filters.")
        return
    
    # Create detailed table
    table_data = []
    for i, complaint in enumerate(filtered_complaints, 1):
        
        # Priority indicator
        category = complaint['priority_category']
        priority_icons = {
            'Critical': '🔴',
            'High': '🟠', 
            'Medium': '🟡',
            'Low': '🟢'
        }
        
        # Status indicator
        status_icons = {
            'submitted': '🆕',
            'under_review': '🔄',
            'completed': '✅',
            'closed': '🔒'
        }
        
        table_data.append({
            'Rank': i,
            'Case ID': complaint['complaint_id'],
            'Title': complaint['incident']['title'][:40] + ('...' if len(complaint['incident']['title']) > 40 else ''),
            'Priority': f"{priority_icons[category]} {complaint['enhanced_priority']:.1f}",
            'Category': category,
            'Original': f"{complaint['priority_score']:.1f}",
            'Type': complaint['incident']['type'],
            'Status': f"{status_icons.get(complaint['status'], '❓')} {complaint['status'].title()}",
            'Age (hrs)': f"{complaint['hours_since_submission']:.1f}",
            'Effort (hrs)': f"{complaint['estimated_effort_hours']:.0f}",
            'Evidence': len(complaint['evidence_files'])
        })
    
    # Display table
    df = pd.DataFrame(table_data)
    
    # Use selection mode for actions
    selected_rows = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun"
    )
    
    # Bulk actions for selected cases
    if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
        st.markdown("### 🔧 Bulk Actions")
        
        selected_indices = selected_rows.selection.rows
        selected_cases = [filtered_complaints[i] for i in selected_indices]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"🤖 Analyze {len(selected_cases)} Cases"):
                st.info("Bulk analysis functionality will be implemented.")
        
        with col2:
            if st.button(f"👤 Assign Cases"):
                show_assignment_dialog(selected_cases)
        
        with col3:
            if st.button(f"📊 Batch Report"):
                st.info("Batch reporting functionality will be implemented.")
        
        with col4:
            if st.button(f"📈 Update Priority"):
                st.info("Priority update functionality will be implemented.")
    
    # Summary statistics for filtered view
    if filtered_complaints:
        st.markdown("---")
        st.markdown("### 📊 Queue Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_priority = sum(c['enhanced_priority'] for c in filtered_complaints) / len(filtered_complaints)
            st.metric("Avg Priority", f"{avg_priority:.1f}")
        
        with col2:
            total_effort = sum(c['estimated_effort_hours'] for c in filtered_complaints)
            st.metric("Total Effort", f"{total_effort:.0f}h")
        
        with col3:
            urgent_cases = len([c for c in filtered_complaints if c['hours_since_submission'] > 24])
            st.metric("Overdue (>24h)", urgent_cases)
        
        with col4:
            avg_age = sum(c['hours_since_submission'] for c in filtered_complaints) / len(filtered_complaints)
            st.metric("Avg Age", f"{avg_age:.1f}h")

def show_assignment_dialog(selected_cases):
    """Show assignment dialog for selected cases"""
    
    with st.expander("👤 Assign Cases to Officer", expanded=True):
        st.markdown(f"**Selected Cases:** {len(selected_cases)}")
        
        # Officer selection
        officers = ["Officer Smith", "Officer Johnson", "Officer Brown", "Officer Davis", "Officer Wilson"]
        selected_officer = st.selectbox("Select Officer:", officers)
        
        # Assignment notes
        assignment_notes = st.text_area("Assignment Notes (optional):")
        
        # Priority override
        override_priority = st.checkbox("Override priority based on officer expertise")
        
        if override_priority:
            new_priority = st.slider("New Priority Level", 1.0, 10.0, 5.0, 0.1)
        
        # Assignment buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Assign Cases", type="primary"):
                # Update selected cases
                for case in selected_cases:
                    case['assigned_officer'] = selected_officer
                    case['assignment_notes'] = assignment_notes
                    case['assignment_date'] = datetime.now().isoformat()
                    if override_priority:
                        case['enhanced_priority'] = new_priority
                
                st.success(f"Successfully assigned {len(selected_cases)} cases to {selected_officer}")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel"):
                st.rerun()
