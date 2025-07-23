"""
Submit Complaint Page - ForensiQ User Interface
==============================================
Allows public users to submit digital forensics complaints with evidence
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import os
import shutil
import hashlib

def show_submit_complaint_page():
    st.markdown('<div class="main-header"><h2>📝 Submit Digital Forensics Complaint</h2><p>Upload evidence and provide details for investigation</p></div>', unsafe_allow_html=True)
    
    # Complaint submission form
    with st.form("complaint_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Basic information
            st.markdown("### 👤 Contact Information")
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
            phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
            
            # Complaint details
            st.markdown("### 📋 Complaint Details")
            complaint_title = st.text_input("Complaint Title *", placeholder="Brief description of the incident")
            
            incident_type = st.selectbox("Incident Type *", [
                "Select incident type...",
                "🔒 Data Breach",
                "💻 Malware/Ransomware",
                "🎣 Phishing Attack", 
                "💳 Financial Fraud",
                "👤 Identity Theft",
                "📱 Mobile Device Compromise",
                "🌐 Network Intrusion",
                "📧 Email Compromise",
                "💾 Data Loss/Corruption",
                "🕵️ Insider Threat",
                "🔗 Social Engineering",
                "🚨 Other Cybercrime"
            ])
            
            urgency_level = st.selectbox("Urgency Level *", [
                "Select urgency level...",
                "🟢 Low - No immediate threat",
                "🟡 Medium - Moderate concern",
                "🟠 High - Significant threat",
                "🔴 Critical - Immediate action required"
            ])
            
            description = st.text_area("Detailed Description *", 
                                     placeholder="Describe the incident in detail. Include timeline, affected systems, suspected cause, and any other relevant information...",
                                     height=150)
            
            # Timeline information
            st.markdown("### ⏰ Incident Timeline")
            incident_date = st.date_input("When did the incident occur?", value=datetime.now().date())
            incident_time = st.time_input("Approximate time", value=datetime.now().time())
            
            discovered_date = st.date_input("When was it discovered?", value=datetime.now().date())
            
        with col2:
            # Evidence upload section
            st.markdown("### 📎 Evidence Upload")
            st.markdown("Upload relevant files as evidence:")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['json', 'txt', 'csv', 'pdf', 'docx', 'xlsx', 'log', 'xml', 'eml', 'msg'],
                help="Supported: .json, .txt, .csv, .pdf, .docx, .xlsx, .log, .xml, .eml, .msg"
            )
            
            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
                total_size = 0
                for file in uploaded_files:
                    file_size = len(file.getvalue())
                    total_size += file_size
                    size_mb = file_size / (1024 * 1024)
                    st.markdown(f"📄 {file.name} ({size_mb:.2f} MB)")
                
                if total_size > 50 * 1024 * 1024:  # 50MB limit
                    st.warning("⚠️ Total file size exceeds 50MB limit")
            
            # Additional information
            st.markdown("### ℹ️ Additional Information")
            
            affected_systems = st.text_area("Affected Systems", 
                                           placeholder="List affected computers, servers, devices...",
                                           height=80)
            
            potential_suspects = st.text_area("Potential Suspects/Sources",
                                            placeholder="Any known or suspected actors...",
                                            height=80)
            
            previous_incidents = st.checkbox("Previous similar incidents occurred")
            
            law_enforcement = st.checkbox("Law enforcement has been notified")
            
            # Privacy consent
            st.markdown("### 🔒 Privacy & Consent")
            consent_analysis = st.checkbox("I consent to automated analysis of uploaded evidence")
            consent_storage = st.checkbox("I consent to secure storage of evidence for investigation")
            privacy_acknowledgment = st.checkbox("I have read and agree to the privacy policy")
        
        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Submit Complaint", use_container_width=True)
        
        if submitted:
            # Validation
            if not all([name, email, complaint_title, description]):
                st.error("❌ Please fill in all required fields (*)")
                return
            
            if incident_type.startswith("Select"):
                st.error("❌ Please select an incident type")
                return
                
            if urgency_level.startswith("Select"):
                st.error("❌ Please select an urgency level")
                return
            
            if not all([consent_analysis, consent_storage, privacy_acknowledgment]):
                st.error("❌ Please accept all privacy and consent requirements")
                return
            
            # Generate complaint ID
            complaint_id = generate_complaint_id()
            
            # Save uploaded files
            evidence_files = []
            if uploaded_files:
                complaint_dir = Path(f"data/raw/complaints/{complaint_id}")
                complaint_dir.mkdir(parents=True, exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    file_path = complaint_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    evidence_files.append({
                        'filename': uploaded_file.name,
                        'size': len(uploaded_file.getvalue()),
                        'type': uploaded_file.type,
                        'path': str(file_path)
                    })
            
            # Create complaint record
            complaint_data = {
                'complaint_id': complaint_id,
                'timestamp': datetime.now().isoformat(),
                'submitter': {
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'username': st.session_state.username
                },
                'incident': {
                    'title': complaint_title,
                    'type': incident_type,
                    'urgency_level': urgency_level,
                    'description': description,
                    'incident_date': incident_date.isoformat(),
                    'incident_time': incident_time.isoformat(),
                    'discovered_date': discovered_date.isoformat(),
                    'affected_systems': affected_systems,
                    'potential_suspects': potential_suspects,
                    'previous_incidents': previous_incidents,
                    'law_enforcement_notified': law_enforcement
                },
                'evidence_files': evidence_files,
                'status': 'submitted',
                'priority_score': calculate_priority_score(incident_type, urgency_level, len(evidence_files)),
                'analysis_status': 'pending',
                'assigned_officer': None,
                'notes': []
            }
            
            # Add to complaints database
            st.session_state.complaints_db.append(complaint_data)
            save_complaints_database()
            
            # Success message
            st.success(f"""
            ✅ **Complaint Successfully Submitted!**
            
            **Complaint ID:** `{complaint_id}`
            
            📧 **Next Steps:**
            - You will receive an email confirmation shortly
            - A cyber security officer will review your case within 24-48 hours
            - You can track your complaint status in the "My Complaints" section
            - Keep your Complaint ID for reference
            """)
            
            # Auto-trigger basic analysis if files were uploaded
            if evidence_files:
                st.info("🤖 Automated analysis has been queued for your evidence files")

def generate_complaint_id():
    """Generate unique complaint ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
    return f"CASE-{timestamp}-{hash_suffix.upper()}"

def calculate_priority_score(incident_type, urgency_level, num_files):
    """Calculate priority score based on incident characteristics"""
    score = 0
    
    # Base score from incident type
    type_scores = {
        'Data Breach': 8,
        'Malware/Ransomware': 9,
        'Financial Fraud': 7,
        'Network Intrusion': 8,
        'Insider Threat': 6,
        'Phishing Attack': 5,
        'Identity Theft': 6,
        'Email Compromise': 4,
        'Mobile Device Compromise': 5,
        'Data Loss/Corruption': 4,
        'Social Engineering': 3,
        'Other Cybercrime': 3
    }
    
    for incident, base_score in type_scores.items():
        if incident in incident_type:
            score += base_score
            break
    
    # Urgency multiplier
    if 'Critical' in urgency_level:
        score *= 1.5
    elif 'High' in urgency_level:
        score *= 1.3
    elif 'Medium' in urgency_level:
        score *= 1.1
    
    # Evidence bonus
    score += min(num_files * 0.5, 3)  # Max 3 points for evidence
    
    return min(round(score, 1), 10)  # Cap at 10

def save_complaints_database():
    """Save complaints to the database file"""
    db_path = Path('data/complaints_db.json')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, 'w') as f:
        json.dump(st.session_state.complaints_db, f, indent=2, default=str)
