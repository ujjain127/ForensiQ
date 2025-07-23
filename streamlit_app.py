"""
ForensiQ Production-Grade Streamlit Dashboard
============================================
Multi-page automated digital forensics platform with role-based access
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime
import hashlib

# Configure page
st.set_page_config(
    page_title="ForensiQ - Digital Forensics Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    /* Professional dark theme base */
    .main {
        background-color: #0f1419;
        color: #ffffff;
        padding-top: 0rem;
    }
    
    .stApp {
        background-color: #0f1419;
        color: #ffffff;
    }
    
    /* Professional Navigation Bar */
    .navbar {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem 2rem;
        border-bottom: 2px solid #475569;
        margin-bottom: 2rem;
        border-radius: 0 0 16px 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .navbar-brand {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-right: 2rem;
        display: inline-block;
    }
    
    .navbar-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
    }
    
    .nav-links {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    
    .nav-link {
        color: #cbd5e0;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
        font-size: 0.9rem;
        border: 1px solid transparent;
    }
    
    .nav-link:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-decoration: none;
    }
    
    .nav-link.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 1px solid #8b5cf6;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .navbar-user {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .user-info {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    .user-name {
        color: #ffffff;
        font-weight: 600;
    }
    
    .user-role {
        color: #8b5cf6;
        font-size: 0.8rem;
        background: rgba(139, 92, 246, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    .logout-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .logout-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    /* Mobile responsive navbar */
    @media (max-width: 768px) {
        .navbar {
            padding: 1rem;
        }
        
        .nav-links {
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .navbar-nav {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    
    /* Professional header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Professional metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #475569;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    /* Stats cards with gradients */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 0.5rem;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .stats-label {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Risk indicators with colors */
    .priority-critical, .priority-high {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3);
    }
    
    .priority-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Login container */
    .login-container {
        max-width: 450px;
        margin: 4rem auto;
        padding: 3rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        border: 1px solid #475569;
        backdrop-filter: blur(10px);
    }
    
    /* Professional complaint cards */
    .complaint-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
        border: 1px solid #475569;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .complaint-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    /* Status indicators */
    .status-submitted {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #000;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .status-under_review, .status-processing {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .status-completed {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .status-closed {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Timeline styling */
    .timeline-event {
        border-left: 3px solid #667eea;
        padding-left: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-left: 1rem;
    }
    
    .timeline-event::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 1.5rem;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 0 0 3px #0f1419;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        color: #ffffff;
        padding: 0.75rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3748;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a202c;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'complaints_db' not in st.session_state:
        st.session_state.complaints_db = load_complaints_database()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'case_overview'

# Load complaints database
def load_complaints_database():
    """Load complaints from the database file or sample data"""
    # First try to load from persistent database
    db_path = Path('data/complaints_db.json')
    if db_path.exists():
        try:
            with open(db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading database: {e}")
    
    # If no database exists, load sample data
    sample_path = Path('data/sample/sample_complaints.json')
    if sample_path.exists():
        try:
            with open(sample_path, 'r') as f:
                sample_data = json.load(f)
                st.success(f"✅ Loaded {len(sample_data)} sample complaints for demonstration")
                return sample_data
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    # Return empty list if nothing found
    st.info("Starting with empty complaint database")
    return []

# Save complaints database
def save_complaints_database():
    """Save complaints to the database file"""
    db_path = Path('data/complaints_db.json')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, 'w') as f:
        json.dump(st.session_state.complaints_db, f, indent=2, default=str)

# Authentication system
def authenticate_user(username, password):
    """Simple authentication system for demo purposes"""
    users = {
        # Public users
        'john_doe': {'password': 'user123', 'role': 'user', 'name': 'John Doe'},
        'jane_smith': {'password': 'user456', 'role': 'user', 'name': 'Jane Smith'},
        'alex_user': {'password': 'demo123', 'role': 'user', 'name': 'Alex Johnson'},
        
        # Cyber Security Officers
        'officer_smith': {'password': 'officer123', 'role': 'officer', 'name': 'Officer Smith'},
        'analyst_jones': {'password': 'analyst456', 'role': 'officer', 'name': 'Analyst Jones'},
        'admin': {'password': 'admin789', 'role': 'officer', 'name': 'System Administrator'}
    }
    
    if username in users and users[username]['password'] == password:
        return users[username]
    return None

# Generate complaint ID
def generate_complaint_id():
    """Generate unique complaint ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
    return f"CASE-{timestamp}-{hash_suffix.upper()}"

# Professional Navigation Bar
def show_navbar():
    """Display professional navigation bar"""
    if not st.session_state.authenticated:
        return
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'case_overview' if st.session_state.user_role == 'officer' else 'submit_complaint')
    
    # Define navigation items based on user role
    if st.session_state.user_role == 'user':
        nav_items = [
            ('🏠', 'Dashboard', 'submit_complaint'),
            ('📝', 'Submit Case', 'submit_complaint'), 
            ('📋', 'My Cases', 'my_complaints'),
            ('❓', 'Help', 'help')
        ]
    else:  # officer role
        nav_items = [
            ('📊', 'Overview', 'case_overview'),
            ('🤖', 'AI Analysis', 'auto_analysis'),
            ('⚠️', 'Priority', 'case_prioritization'),
            ('⏱️', 'Timeline', 'timeline_reconstruction'),
            ('🕸️', 'Network', 'actor_graph'),
            ('📊', 'Analytics', 'system_analytics')
        ]
    
    # Create navbar using columns for better control
    st.markdown("""
    <div class="navbar">
        <div class="navbar-nav">
            <div class="navbar-brand">🔍 ForensiQ</div>
        </div>
        <div class="navbar-user">
            <div class="user-info">
                <span class="user-name">{}</span>
                <div class="user-role">{}</div>
            </div>
        </div>
    </div>
    """.format(st.session_state.user_name, st.session_state.user_role.title()), unsafe_allow_html=True)
    
    # Create navigation using columns
    nav_cols = st.columns(len(nav_items) + 1)  # +1 for logout
    
    for i, (icon, label, page_key) in enumerate(nav_items):
        with nav_cols[i]:
            if st.button(f"{icon} {label}", key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
    
    # Logout button in last column
    with nav_cols[-1]:
        if st.button("🚪 Logout", key="nav_logout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.session_state.user_name = None
            st.rerun()
    
    # Style the navigation buttons to look like navbar
    st.markdown("""
    <style>
    /* Style navigation buttons to look like navbar links */
    .element-container:has(.stButton) {
        margin-top: -2rem;
    }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:active,
    .stButton > button:focus {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(0);
    }
    </style>
    """, unsafe_allow_html=True)

# Login page
def show_login_page():
    # Professional header with gradient
    st.markdown("""
    <div class="main-header">
        <h1 class="app-title">🔍 ForensiQ</h1>
        <p style="font-size: 1.2rem; opacity: 0.9; margin: 0;">Digital Forensics Intelligence Platform</p>
        <p style="font-size: 1rem; opacity: 0.8; margin-top: 0.5rem;">Advanced AI-Powered Cyber Security Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 0.5rem;">🔐 Secure Access</h2>
                <p style="color: #94a3b8; margin: 0;">Enter your credentials to continue</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username", help="Use demo credentials below")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_login, col_clear = st.columns([3, 1])
            with col_login:
                login_button = st.form_submit_button("🚀 Login to ForensiQ", use_container_width=True)
            with col_clear:
                if st.form_submit_button("Clear"):
                    st.rerun()
            
            if login_button:
                if username and password:
                    user_data = authenticate_user(username, password)
                    if user_data:
                        st.session_state.authenticated = True
                        st.session_state.user_role = user_data['role']
                        st.session_state.username = username
                        st.session_state.user_name = user_data['name']
                        st.balloons()
                        st.success(f"✅ Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Access denied. Invalid credentials.")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Professional demo credentials section
        st.markdown("---")
        
        # Create professional credential cards
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="text-align: center; color: #ffffff; margin-bottom: 1.5rem;">🎯 Demo Access Credentials</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_user, col_officer = st.columns(2)
        
        with col_user:
            st.markdown("""
            <div class="stats-card" style="background: linear-gradient(135deg, #10b981 0%, #047857 100%);">
                <h4 style="margin: 0 0 1rem 0; font-size: 1.1rem;">👤 Public Users</h4>
                <div style="text-align: left; font-family: monospace; font-size: 0.9rem;">
                    <strong>Username:</strong> john_doe<br>
                    <strong>Password:</strong> user123<br><br>
                    <strong>Username:</strong> jane_smith<br>
                    <strong>Password:</strong> user456
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_officer:
            st.markdown("""
            <div class="stats-card" style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);">
                <h4 style="margin: 0 0 1rem 0; font-size: 1.1rem;">👮 Cyber Officers</h4>
                <div style="text-align: left; font-family: monospace; font-size: 0.9rem;">
                    <strong>Username:</strong> officer_smith<br>
                    <strong>Password:</strong> officer123<br><br>
                    <strong>Username:</strong> admin<br>
                    <strong>Password:</strong> admin789
                </div>
            </div>
            """, unsafe_allow_html=True)

# Sidebar navigation
def show_sidebar():
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.user_name}")
        st.markdown(f"**Role:** {st.session_state.user_role.title()}")
        
        st.markdown("---")
        
        if st.session_state.user_role == 'user':
            # Public user navigation
            st.markdown("### 📝 User Dashboard")
            pages = {
                "Submit Complaint": "submit_complaint",
                "My Complaints": "my_complaints",
                "Help & Guidelines": "help"
            }
        else:
            # Officer navigation
            st.markdown("### 🛡️ Officer Dashboard")
            pages = {
                "Case Overview": "case_overview",
                "Auto Analysis": "auto_analysis",
                "Case Prioritization": "case_prioritization",
                "Timeline Reconstruction": "timeline_reconstruction",
                "Actor-Resource Graph": "actor_graph",
                "System Analytics": "system_analytics"
            }
        
        # Navigation buttons
        for page_name, page_key in pages.items():
            if st.button(f"📊 {page_name}", key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🔧 System Status")
        st.markdown("🟢 **Analysis Engine:** Online")
        st.markdown("🟢 **Database:** Connected")
        st.markdown(f"🟢 **Cases:** {len(st.session_state.complaints_db)}")
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Main app logic
def main():
    load_css()
    initialize_session()
    
    # Check authentication
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Show professional navbar instead of sidebar
    show_navbar()
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        if st.session_state.user_role == 'user':
            st.session_state.current_page = 'submit_complaint'
        else:
            st.session_state.current_page = 'case_overview'
    
    # Route to appropriate page based on role and selection
    current_page = st.session_state.current_page
    
    if st.session_state.user_role == 'user':
        if current_page == 'submit_complaint':
            from pages.submit_complaint import show_submit_complaint_page
            show_submit_complaint_page()
        elif current_page == 'my_complaints':
            from pages.my_complaints import show_my_complaints_page
            show_my_complaints_page()
        elif current_page == 'help':
            from pages.help import show_help_page
            show_help_page()
    
    else:  # Officer role
        if current_page == 'case_overview':
            from pages.case_overview import show_case_overview_page
            show_case_overview_page()
        elif current_page == 'auto_analysis':
            from pages.auto_analysis import show_auto_analysis_page
            show_auto_analysis_page()
        elif current_page == 'case_prioritization':
            from pages.case_prioritization import show_case_prioritization_page
            show_case_prioritization_page()
        elif current_page == 'timeline_reconstruction':
            from pages.timeline_reconstruction import show_timeline_reconstruction_page
            show_timeline_reconstruction_page()
        elif current_page == 'actor_graph':
            from pages.actor_graph import show_actor_graph_page
            show_actor_graph_page()
        elif current_page == 'system_analytics':
            from pages.system_analytics import show_system_analytics_page
            show_system_analytics_page()

if __name__ == "__main__":
    main()
