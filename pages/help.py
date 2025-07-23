"""
Help Page - ForensiQ User Guide
===============================
Comprehensive help and documentation for ForensiQ users
"""

import streamlit as st

def show_help_page():
    st.markdown("""
    <div class="main-header">
        <h1 class="app-title">❓ Help & User Guide</h1>
        <p style="font-size: 1.1rem; opacity: 0.9; margin: 0;">Complete guide to using ForensiQ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Help sections
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Getting Started", "📝 Submitting Cases", "📊 Understanding Results", "🔧 Troubleshooting"])
    
    with tab1:
        st.markdown("""
        <div class="complaint-card">
            <h3 style="color: #8b5cf6;">🚀 Getting Started with ForensiQ</h3>
            
            <h4 style="color: #60a5fa;">What is ForensiQ?</h4>
            <p>ForensiQ is an advanced digital forensics platform that helps users report and investigate cybersecurity incidents using AI-powered analysis.</p>
            
            <h4 style="color: #60a5fa;">User Roles</h4>
            <ul>
                <li><strong>Public Users:</strong> Submit complaints and track case status</li>
                <li><strong>Security Officers:</strong> Investigate cases and perform analysis</li>
            </ul>
            
            <h4 style="color: #60a5fa;">Key Features</h4>
            <ul>
                <li>🤖 AI-powered incident analysis</li>
                <li>📊 Real-time case tracking</li>
                <li>🔍 Evidence processing and analysis</li>
                <li>📈 Timeline reconstruction</li>
                <li>🕸️ Network relationship mapping</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="complaint-card">
            <h3 style="color: #8b5cf6;">📝 How to Submit a Case</h3>
            
            <h4 style="color: #60a5fa;">Step 1: Choose Incident Type</h4>
            <p>Select the type of cybersecurity incident you're reporting:</p>
            <ul>
                <li><strong>Phishing:</strong> Suspicious emails or websites</li>
                <li><strong>Malware:</strong> Virus or malicious software</li>
                <li><strong>Data Breach:</strong> Unauthorized access to data</li>
                <li><strong>Identity Theft:</strong> Stolen personal information</li>
                <li><strong>Financial Fraud:</strong> Unauthorized financial transactions</li>
                <li><strong>Ransomware:</strong> Files encrypted by attackers</li>
                <li><strong>Social Engineering:</strong> Manipulation tactics</li>
                <li><strong>Other:</strong> Any other cybersecurity incident</li>
            </ul>
            
            <h4 style="color: #60a5fa;">Step 2: Provide Details</h4>
            <ul>
                <li>Write a clear, detailed description</li>
                <li>Include timeline of events</li>
                <li>Mention any financial impact</li>
                <li>Note any suspicious indicators</li>
            </ul>
            
            <h4 style="color: #60a5fa;">Step 3: Upload Evidence</h4>
            <p>Attach relevant files such as:</p>
            <ul>
                <li>Screenshots of suspicious activity</li>
                <li>Email headers or message sources</li>
                <li>Log files or system outputs</li>
                <li>Any other relevant documentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="complaint-card">
            <h3 style="color: #8b5cf6;">📊 Understanding Your Case Results</h3>
            
            <h4 style="color: #60a5fa;">Case Status</h4>
            <ul>
                <li><span class="status-submitted">Submitted</span> - Case received and queued</li>
                <li><span class="status-under_review">Under Review</span> - Being analyzed by our team</li>
                <li><span class="status-completed">Completed</span> - Investigation finished</li>
                <li><span class="status-closed">Closed</span> - Case resolved</li>
            </ul>
            
            <h4 style="color: #60a5fa;">Priority Levels</h4>
            <ul>
                <li><span class="priority-critical">Critical</span> - Immediate attention required</li>
                <li><span class="priority-high">High</span> - Urgent investigation needed</li>
                <li><span class="priority-medium">Medium</span> - Standard priority</li>
                <li><span class="priority-low">Low</span> - Lower priority case</li>
            </ul>
            
            <h4 style="color: #60a5fa;">AI Analysis Results</h4>
            <p>Our AI system provides:</p>
            <ul>
                <li><strong>Entity Extraction:</strong> Key people, organizations, locations</li>
                <li><strong>Risk Assessment:</strong> Threat level evaluation</li>
                <li><strong>Pattern Detection:</strong> Similar incident identification</li>
                <li><strong>Timeline Analysis:</strong> Event sequence reconstruction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="complaint-card">
            <h3 style="color: #8b5cf6;">🔧 Troubleshooting & FAQ</h3>
            
            <h4 style="color: #60a5fa;">Common Issues</h4>
            
            <div style="margin: 1rem 0;">
                <strong>Q: I can't upload my evidence file</strong><br>
                <strong>A:</strong> Ensure your file is under 200MB and in a supported format (PDF, DOC, TXT, PNG, JPG).
            </div>
            
            <div style="margin: 1rem 0;">
                <strong>Q: My case status hasn't updated</strong><br>
                <strong>A:</strong> Cases are processed in order of priority. Check back in 24-48 hours for updates.
            </div>
            
            <div style="margin: 1rem 0;">
                <strong>Q: How long does investigation take?</strong><br>
                <strong>A:</strong> Investigation time varies by complexity. Simple cases: 1-3 days, Complex cases: 1-2 weeks.
            </div>
            
            <div style="margin: 1rem 0;">
                <strong>Q: Can I update my case after submission?</strong><br>
                <strong>A:</strong> Contact our support team with your case ID to add additional information.
            </div>
            
            <h4 style="color: #60a5fa;">Contact Support</h4>
            <p>If you need additional help:</p>
            <ul>
                <li>📧 Email: support@forensiq.com</li>
                <li>📞 Phone: 1-800-FORENSIQ</li>
                <li>💬 Live Chat: Available 24/7</li>
            </ul>
            
            <h4 style="color: #60a5fa;">Emergency Contacts</h4>
            <p>For critical security incidents:</p>
            <ul>
                <li>🚨 Emergency Hotline: 1-800-CYBER-911</li>
                <li>🏛️ Report to FBI IC3: www.ic3.gov</li>
                <li>🛡️ CISA: www.cisa.gov/report</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Submit New Case", use_container_width=True):
            st.session_state.current_page = 'submit_complaint'
            st.rerun()
    
    with col2:
        if st.button("📋 View My Cases", use_container_width=True):
            st.session_state.current_page = 'my_complaints'
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <a href="mailto:support@forensiq.com" style="color: #8b5cf6; text-decoration: none;">
                📧 Contact Support
            </a>
        </div>
        """, unsafe_allow_html=True)
