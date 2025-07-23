"""
My Complaints Page - ForensiQ User Interface
===========================================
Shows user's submitted complaints and their status
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

def show_my_complaints_page():
    st.markdown('<div class="main-header"><h2>📋 My Complaints</h2><p>Track the status of your submitted complaints</p></div>', unsafe_allow_html=True)
    
    # Get user's complaints
    user_complaints = [complaint for complaint in st.session_state.complaints_db 
                      if complaint['submitter']['username'] == st.session_state.username]
    
    if not user_complaints:
        st.markdown("### 📭 No Complaints Found")
        st.info("You haven't submitted any complaints yet. Use the 'Submit Complaint' page to file your first case.")
        
        if st.button("📝 Submit New Complaint"):
            st.session_state.current_page = 'submit_complaint'
            st.rerun()
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Complaints", len(user_complaints))
    
    with col2:
        pending_count = len([c for c in user_complaints if c['status'] in ['submitted', 'under_review']])
        st.metric("Pending", pending_count)
    
    with col3:
        completed_count = len([c for c in user_complaints if c['status'] == 'completed'])
        st.metric("Completed", completed_count)
    
    with col4:
        avg_priority = sum(c['priority_score'] for c in user_complaints) / len(user_complaints)
        st.metric("Avg Priority", f"{avg_priority:.1f}")
    
    st.markdown("---")
    
    # Filters and sorting
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search complaints", placeholder="Search by title, type, or ID...")
    
    with col2:
        status_filter = st.selectbox("Filter by Status", ["All", "Submitted", "Under Review", "Completed", "Closed"])
    
    with col3:
        sort_option = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Priority (High)", "Priority (Low)"])
    
    # Filter complaints
    filtered_complaints = user_complaints.copy()
    
    if search_term:
        filtered_complaints = [
            c for c in filtered_complaints 
            if search_term.lower() in c['complaint_id'].lower() 
            or search_term.lower() in c['incident']['title'].lower()
            or search_term.lower() in c['incident']['type'].lower()
        ]
    
    if status_filter != "All":
        status_map = {
            "Submitted": "submitted",
            "Under Review": "under_review", 
            "Completed": "completed",
            "Closed": "closed"
        }
        filtered_complaints = [c for c in filtered_complaints if c['status'] == status_map[status_filter]]
    
    # Sort complaints
    if sort_option == "Date (Newest)":
        filtered_complaints.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_option == "Date (Oldest)":
        filtered_complaints.sort(key=lambda x: x['timestamp'])
    elif sort_option == "Priority (High)":
        filtered_complaints.sort(key=lambda x: x['priority_score'], reverse=True)
    elif sort_option == "Priority (Low)":
        filtered_complaints.sort(key=lambda x: x['priority_score'])
    
    # Display complaints
    for complaint in filtered_complaints:
        display_complaint_card(complaint)

def display_complaint_card(complaint):
    """Display a complaint card with details and status"""
    
    # Determine status styling
    status = complaint['status']
    status_styles = {
        'submitted': ('🟡', 'status-new', 'Submitted'),
        'under_review': ('🔄', 'status-processing', 'Under Review'),
        'completed': ('✅', 'status-completed', 'Completed'),
        'closed': ('🔒', 'status-closed', 'Closed')
    }
    
    status_icon, status_class, status_text = status_styles.get(status, ('❓', 'status-new', 'Unknown'))
    
    # Priority styling
    priority = complaint['priority_score']
    if priority >= 7:
        priority_class = 'priority-high'
        priority_icon = '🔴'
    elif priority >= 4:
        priority_class = 'priority-medium'
        priority_icon = '🟡'
    else:
        priority_class = 'priority-low' 
        priority_icon = '🟢'
    
    # Create complaint card
    with st.container():
        st.markdown(f"""
        <div class="complaint-card {priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #1e3c72;">
                    {priority_icon} {complaint['incident']['title']}
                </h4>
                <span class="status-indicator {status_class}">
                    {status_icon} {status_text}
                </span>
            </div>
            
            <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
                <div><strong>Case ID:</strong> {complaint['complaint_id']}</div>
                <div><strong>Type:</strong> {complaint['incident']['type']}</div>
                <div><strong>Priority:</strong> {priority}/10</div>
                <div><strong>Submitted:</strong> {datetime.fromisoformat(complaint['timestamp']).strftime('%Y-%m-%d %H:%M')}</div>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <strong>Description:</strong> {complaint['incident']['description'][:150]}{'...' if len(complaint['incident']['description']) > 150 else ''}
            </div>
            
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <div><strong>Evidence Files:</strong> {len(complaint['evidence_files'])}</div>
                <div><strong>Analysis:</strong> {complaint.get('analysis_status', 'pending').title()}</div>
                {f"<div><strong>Assigned:</strong> {complaint.get('assigned_officer', 'Unassigned')}</div>" if complaint.get('assigned_officer') else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            if st.button("🔍 View Details", key=f"view_{complaint['complaint_id']}"):
                show_complaint_details(complaint)
        
        with col2:
            if complaint['evidence_files'] and st.button("📁 Evidence", key=f"evidence_{complaint['complaint_id']}"):
                show_evidence_details(complaint)
        
        with col3:
            if complaint.get('analysis_status') == 'completed' and st.button("📊 Report", key=f"report_{complaint['complaint_id']}"):
                show_analysis_report(complaint)
        
        st.markdown("---")

def show_complaint_details(complaint):
    """Show detailed complaint information in an expander"""
    with st.expander(f"📋 Complaint Details - {complaint['complaint_id']}", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Incident Information")
            st.markdown(f"**Title:** {complaint['incident']['title']}")
            st.markdown(f"**Type:** {complaint['incident']['type']}")
            st.markdown(f"**Urgency:** {complaint['incident']['urgency_level']}")
            st.markdown(f"**Priority Score:** {complaint['priority_score']}/10")
            
            st.markdown("### ⏰ Timeline")
            st.markdown(f"**Incident Date:** {complaint['incident']['incident_date']}")
            st.markdown(f"**Discovered Date:** {complaint['incident']['discovered_date']}")
            st.markdown(f"**Submitted:** {datetime.fromisoformat(complaint['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            
        with col2:
            st.markdown("### 🛡️ Investigation Status")
            st.markdown(f"**Status:** {complaint['status'].title()}")
            st.markdown(f"**Analysis:** {complaint.get('analysis_status', 'pending').title()}")
            st.markdown(f"**Assigned Officer:** {complaint.get('assigned_officer', 'Unassigned')}")
            
            if complaint.get('notes'):
                st.markdown("### 📝 Investigation Notes")
                for note in complaint['notes'][-3:]:  # Show last 3 notes
                    st.markdown(f"- {note}")
        
        st.markdown("### 📋 Full Description")
        st.markdown(complaint['incident']['description'])
        
        if complaint['incident']['affected_systems']:
            st.markdown("### 💻 Affected Systems")
            st.markdown(complaint['incident']['affected_systems'])

def show_evidence_details(complaint):
    """Show evidence file details"""
    with st.expander(f"📁 Evidence Files - {complaint['complaint_id']}", expanded=True):
        
        if not complaint['evidence_files']:
            st.info("No evidence files uploaded for this complaint.")
            return
        
        st.markdown(f"### 📎 {len(complaint['evidence_files'])} Evidence File(s)")
        
        # Create evidence table
        evidence_data = []
        for evidence in complaint['evidence_files']:
            size_mb = evidence['size'] / (1024 * 1024)
            evidence_data.append({
                'Filename': evidence['filename'],
                'Type': evidence.get('type', 'Unknown'),
                'Size (MB)': f"{size_mb:.2f}",
                'Status': '✅ Uploaded'
            })
        
        df = pd.DataFrame(evidence_data)
        st.dataframe(df, use_container_width=True)
        
        # Show file analysis status if available
        if complaint.get('analysis_results'):
            st.markdown("### 🤖 Analysis Results Preview")
            analysis = complaint['analysis_results']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'classification' in analysis:
                    classification = analysis['classification']
                    if classification == 'malicious':
                        st.error(f"🚨 Classification: {classification.title()}")
                    elif classification == 'suspicious':
                        st.warning(f"⚠️ Classification: {classification.title()}")
                    else:
                        st.success(f"✅ Classification: {classification.title()}")
            
            with col2:
                if 'entities_found' in analysis:
                    st.metric("Entities Found", len(analysis['entities_found']))
            
            with col3:
                if 'threat_score' in analysis:
                    threat_score = analysis['threat_score']
                    st.metric("Threat Score", f"{threat_score}/10")

def show_analysis_report(complaint):
    """Show analysis report for completed cases"""
    with st.expander(f"📊 Analysis Report - {complaint['complaint_id']}", expanded=True):
        
        if complaint.get('analysis_status') != 'completed':
            st.warning("Analysis is still in progress or has not been started.")
            return
        
        analysis = complaint.get('analysis_results', {})
        
        if not analysis:
            st.info("Analysis results are not yet available.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            classification = analysis.get('classification', 'unknown')
            if classification == 'malicious':
                st.error(f"🚨 {classification.title()}")
            elif classification == 'suspicious':
                st.warning(f"⚠️ {classification.title()}")
            else:
                st.success(f"✅ {classification.title()}")
        
        with col2:
            st.metric("Threat Score", f"{analysis.get('threat_score', 0)}/10")
        
        with col3:
            st.metric("Entities Found", len(analysis.get('entities_found', [])))
        
        with col4:
            st.metric("Files Analyzed", len(analysis.get('file_analysis', [])))
        
        # Detailed findings
        if analysis.get('entities_found'):
            st.markdown("### 🎯 Extracted Entities")
            entities_df = pd.DataFrame(analysis['entities_found'])
            st.dataframe(entities_df, use_container_width=True)
        
        if analysis.get('sentiment_analysis'):
            st.markdown("### 😊 Sentiment Analysis")
            sentiment = analysis['sentiment_analysis']
            st.markdown(f"**Urgency Level:** {sentiment.get('urgency', 'Unknown')}")
            st.markdown(f"**Emotional Tone:** {sentiment.get('emotion', 'Unknown')}")
        
        if analysis.get('recommendations'):
            st.markdown("### 💡 Recommendations")
            for rec in analysis['recommendations']:
                st.markdown(f"- {rec}")
        
        # Download report button
        if st.button("📄 Download Full Report", key=f"download_{complaint['complaint_id']}"):
            st.info("Report download functionality will be implemented in the next update.")
