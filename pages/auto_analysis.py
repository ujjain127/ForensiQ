"""
Auto Analysis Page - ForensiQ Officer Interface
==============================================
Automated ML/NLP analysis of digital evidence and complaints
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import time
import plotly.graph_objects as go
import plotly.express as px

def show_auto_analysis_page():
    st.markdown('<div class="main-header"><h2>🤖 Automated Analysis</h2><p>AI-powered analysis of digital evidence and complaints</p></div>', unsafe_allow_html=True)
    
    # Check if a case is selected
    selected_case = st.session_state.get('selected_case')
    
    if selected_case:
        show_case_analysis(selected_case)
    else:
        show_analysis_dashboard()

def show_analysis_dashboard():
    """Show overall analysis dashboard when no specific case is selected"""
    
    # Quick stats
    all_complaints = st.session_state.complaints_db
    pending_analysis = [c for c in all_complaints if c.get('analysis_status', 'pending') == 'pending']
    completed_analysis = [c for c in all_complaints if c.get('analysis_status') == 'completed']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", len(all_complaints))
    
    with col2:
        st.metric("Pending Analysis", len(pending_analysis))
    
    with col3:
        st.metric("Completed Analysis", len(completed_analysis))
    
    with col4:
        if all_complaints:
            avg_threat = sum(c.get('analysis_results', {}).get('threat_score', 0) for c in completed_analysis) / max(len(completed_analysis), 1)
            st.metric("Avg Threat Score", f"{avg_threat:.1f}/10")
        else:
            st.metric("Avg Threat Score", "0.0/10")
    
    st.markdown("---")
    
    # Case selection for analysis
    st.markdown("### 🎯 Select Case for Analysis")
    
    if not all_complaints:
        st.info("No cases available for analysis. Cases will appear here once users submit complaints.")
        return
    
    # Filter cases that need analysis
    cases_for_analysis = [c for c in all_complaints if c.get('analysis_status', 'pending') in ['pending', 'failed']]
    
    if not cases_for_analysis:
        st.success("✅ All cases have been analyzed!")
        
        # Show recent analysis results
        st.markdown("### 📊 Recent Analysis Results")
        show_recent_analysis_results(completed_analysis[-5:] if completed_analysis else [])
        return
    
    # Case selection
    case_options = [f"{c['complaint_id']} - {c['incident']['title']}" for c in cases_for_analysis]
    selected_option = st.selectbox("Choose a case to analyze:", case_options)
    
    if selected_option:
        selected_idx = case_options.index(selected_option)
        selected_case = cases_for_analysis[selected_idx]
        
        # Display case info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Case ID:** {selected_case['complaint_id']}")
            st.markdown(f"**Title:** {selected_case['incident']['title']}")
            st.markdown(f"**Type:** {selected_case['incident']['type']}")
            st.markdown(f"**Priority:** {selected_case['priority_score']}/10")
            st.markdown(f"**Evidence Files:** {len(selected_case['evidence_files'])}")
        
        with col2:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                run_automated_analysis(selected_case)

def show_case_analysis(selected_case):
    """Show analysis for a specific selected case"""
    
    st.markdown(f"### 🔍 Analysis: {selected_case['complaint_id']}")
    st.markdown(f"**{selected_case['incident']['title']}**")
    
    # Case details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Type:** {selected_case['incident']['type']}")
        st.markdown(f"**Priority:** {selected_case['priority_score']}/10")
    
    with col2:
        st.markdown(f"**Status:** {selected_case['status'].title()}")
        st.markdown(f"**Evidence Files:** {len(selected_case['evidence_files'])}")
    
    with col3:
        analysis_status = selected_case.get('analysis_status', 'pending')
        st.markdown(f"**Analysis Status:** {analysis_status.title()}")
        if selected_case.get('assigned_officer'):
            st.markdown(f"**Assigned:** {selected_case['assigned_officer']}")
    
    st.markdown("---")
    
    # Check analysis status
    analysis_status = selected_case.get('analysis_status', 'pending')
    
    if analysis_status == 'pending':
        st.info("⏳ Analysis has not been started for this case.")
        if st.button("🚀 Start Analysis", type="primary"):
            run_automated_analysis(selected_case)
    
    elif analysis_status == 'running':
        st.warning("🔄 Analysis is currently in progress...")
        show_analysis_progress()
    
    elif analysis_status == 'completed':
        show_completed_analysis(selected_case)
    
    elif analysis_status == 'failed':
        st.error("❌ Analysis failed. Please try again or contact system administrator.")
        if st.button("🔄 Retry Analysis"):
            run_automated_analysis(selected_case)

def run_automated_analysis(case):
    """Run automated ML/NLP analysis on the selected case"""
    
    # Update case status
    case['analysis_status'] = 'running'
    
    st.info("🚀 Starting automated analysis...")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    analysis_steps = [
        ("Initializing analysis engine...", 10),
        ("Processing evidence files...", 25),
        ("Running NLP entity extraction...", 40),
        ("Performing threat classification...", 60),
        ("Analyzing communication patterns...", 75),
        ("Generating insights and recommendations...", 90),
        ("Finalizing results...", 100)
    ]
    
    results = {}
    
    for step_text, progress in analysis_steps:
        status_text.text(step_text)
        progress_bar.progress(progress)
        time.sleep(1.5)  # Simulate processing time
        
        # Simulate different analysis components
        if "entity extraction" in step_text:
            results['entities_found'] = simulate_entity_extraction(case)
        elif "threat classification" in step_text:
            results['classification'] = simulate_threat_classification(case)
            results['threat_score'] = simulate_threat_scoring(case)
        elif "communication patterns" in step_text:
            results['sentiment_analysis'] = simulate_sentiment_analysis(case)
        elif "insights" in step_text:
            results['recommendations'] = simulate_recommendations(case)
    
    # Complete analysis
    case['analysis_status'] = 'completed'
    case['analysis_results'] = results
    case['analysis_completed_at'] = datetime.now().isoformat()
    
    status_text.text("✅ Analysis completed successfully!")
    progress_bar.progress(100)
    
    time.sleep(1)
    st.rerun()

def simulate_entity_extraction(case):
    """Simulate NLP entity extraction"""
    
    # Mock entities based on case type and description
    entities = []
    description = case['incident']['description'].lower()
    
    # Email patterns
    if 'email' in description or '@' in description:
        entities.extend([
            {'type': 'EMAIL', 'value': 'suspicious@fake-bank.com', 'confidence': 0.95},
            {'type': 'EMAIL', 'value': 'victim@company.com', 'confidence': 0.88}
        ])
    
    # IP patterns
    if 'ip' in description or 'address' in description:
        entities.extend([
            {'type': 'IP_ADDRESS', 'value': '192.168.1.100', 'confidence': 0.92},
            {'type': 'IP_ADDRESS', 'value': '10.0.0.50', 'confidence': 0.87}
        ])
    
    # URL patterns
    if 'website' in description or 'link' in description or 'url' in description:
        entities.extend([
            {'type': 'URL', 'value': 'https://fake-banking-site.net', 'confidence': 0.96},
            {'type': 'URL', 'value': 'http://malicious-download.org', 'confidence': 0.89}
        ])
    
    # Phone numbers
    if 'phone' in description or 'call' in description:
        entities.append({'type': 'PHONE', 'value': '+1-555-123-4567', 'confidence': 0.91})
    
    # Organizations
    if 'bank' in description or 'company' in description:
        entities.append({'type': 'ORGANIZATION', 'value': 'First National Bank', 'confidence': 0.85})
    
    # Default entities if none found
    if not entities:
        entities = [
            {'type': 'PERSON', 'value': 'John Suspect', 'confidence': 0.78},
            {'type': 'DATE', 'value': '2024-01-15', 'confidence': 0.82}
        ]
    
    return entities

def simulate_threat_classification(case):
    """Simulate threat classification"""
    
    description = case['incident']['description'].lower()
    incident_type = case['incident']['type'].lower()
    
    # Classification logic
    if any(word in description for word in ['malware', 'virus', 'trojan', 'ransomware']):
        return 'malicious'
    elif any(word in description for word in ['phishing', 'scam', 'fraud', 'suspicious']):
        return 'suspicious'
    elif case['priority_score'] >= 7:
        return 'suspicious'
    else:
        return 'benign'

def simulate_threat_scoring(case):
    """Simulate threat scoring"""
    
    base_score = case['priority_score']
    description = case['incident']['description'].lower()
    
    # Adjust based on keywords
    if any(word in description for word in ['urgent', 'critical', 'emergency']):
        base_score += 1
    
    if any(word in description for word in ['money', 'financial', 'bank', 'payment']):
        base_score += 0.5
    
    if len(case['evidence_files']) > 3:
        base_score += 0.5
    
    return min(10, max(1, round(base_score, 1)))

def simulate_sentiment_analysis(case):
    """Simulate sentiment analysis"""
    
    description = case['incident']['description'].lower()
    
    # Determine urgency
    if any(word in description for word in ['urgent', 'emergency', 'immediately', 'asap']):
        urgency = 'high'
    elif any(word in description for word in ['soon', 'quickly', 'prompt']):
        urgency = 'medium'
    else:
        urgency = 'low'
    
    # Determine emotion
    if any(word in description for word in ['angry', 'frustrated', 'upset']):
        emotion = 'negative'
    elif any(word in description for word in ['worried', 'concerned', 'anxious']):
        emotion = 'concerned'
    else:
        emotion = 'neutral'
    
    return {
        'urgency': urgency,
        'emotion': emotion,
        'confidence': 0.83
    }

def simulate_recommendations(case):
    """Generate recommendations based on analysis"""
    
    recommendations = []
    classification = simulate_threat_classification(case)
    threat_score = simulate_threat_scoring(case)
    
    if classification == 'malicious':
        recommendations.extend([
            "Immediate isolation of affected systems recommended",
            "Conduct full forensic imaging of compromised devices",
            "Reset all potentially compromised credentials",
            "Implement additional monitoring for lateral movement"
        ])
    elif classification == 'suspicious':
        recommendations.extend([
            "Enhanced monitoring of related network traffic",
            "User security awareness training recommended",
            "Review and update security policies",
            "Consider threat hunting activities"
        ])
    else:
        recommendations.extend([
            "Standard security protocols appear sufficient",
            "Continue regular monitoring",
            "Document incident for future reference"
        ])
    
    if threat_score >= 7:
        recommendations.append("Escalate to senior security team")
    
    if len(case['evidence_files']) > 0:
        recommendations.append("Detailed forensic analysis of evidence files recommended")
    
    return recommendations

def show_completed_analysis(case):
    """Display completed analysis results"""
    
    results = case.get('analysis_results', {})
    
    if not results:
        st.error("Analysis results not found.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        classification = results.get('classification', 'unknown')
        if classification == 'malicious':
            st.error(f"🚨 {classification.title()}")
        elif classification == 'suspicious':
            st.warning(f"⚠️ {classification.title()}")
        else:
            st.success(f"✅ {classification.title()}")
    
    with col2:
        threat_score = results.get('threat_score', 0)
        if threat_score >= 7:
            st.error(f"🔴 Threat: {threat_score}/10")
        elif threat_score >= 4:
            st.warning(f"🟡 Threat: {threat_score}/10")
        else:
            st.success(f"🟢 Threat: {threat_score}/10")
    
    with col3:
        entities_count = len(results.get('entities_found', []))
        st.metric("Entities Found", entities_count)
    
    with col4:
        recommendations_count = len(results.get('recommendations', []))
        st.metric("Recommendations", recommendations_count)
    
    st.markdown("---")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Entities", "😊 Sentiment", "💡 Recommendations", "📊 Raw Data"])
    
    with tab1:
        show_entity_analysis(results)
    
    with tab2:
        show_sentiment_analysis_results(results)
    
    with tab3:
        show_recommendations_analysis(results)
    
    with tab4:
        show_raw_analysis_data(results)

def show_entity_analysis(results):
    """Display extracted entities"""
    
    entities = results.get('entities_found', [])
    
    if not entities:
        st.info("No entities were extracted from the analysis.")
        return
    
    # Create entities dataframe
    df = pd.DataFrame(entities)
    
    # Entity type distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Entity Distribution")
        entity_counts = df['type'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=entity_counts.values,
            y=entity_counts.index,
            orientation='h',
            marker_color='#2196F3'
        )])
        
        fig.update_layout(
            xaxis_title="Count",
            yaxis_title="Entity Type",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Confidence Levels")
        
        # Confidence distribution
        df['confidence_range'] = pd.cut(df['confidence'], 
                                       bins=[0, 0.7, 0.85, 1.0], 
                                       labels=['Low', 'Medium', 'High'])
        
        confidence_counts = df['confidence_range'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=confidence_counts.index,
            values=confidence_counts.values,
            hole=0.3
        )])
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed entities table
    st.markdown("#### 📋 Extracted Entities")
    
    # Format confidence as percentage
    df['confidence_pct'] = (df['confidence'] * 100).round(1).astype(str) + '%'
    
    display_df = df[['type', 'value', 'confidence_pct']].copy()
    display_df.columns = ['Type', 'Value', 'Confidence']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def show_sentiment_analysis_results(results):
    """Display sentiment analysis results"""
    
    sentiment = results.get('sentiment_analysis', {})
    
    if not sentiment:
        st.info("Sentiment analysis data not available.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        urgency = sentiment.get('urgency', 'unknown')
        if urgency == 'high':
            st.error(f"🚨 Urgency: {urgency.title()}")
        elif urgency == 'medium':
            st.warning(f"⚠️ Urgency: {urgency.title()}")
        else:
            st.success(f"✅ Urgency: {urgency.title()}")
    
    with col2:
        emotion = sentiment.get('emotion', 'unknown')
        emotion_icons = {
            'negative': '😠',
            'concerned': '😟',
            'neutral': '😐',
            'positive': '😊'
        }
        icon = emotion_icons.get(emotion, '❓')
        st.metric("Emotion", f"{icon} {emotion.title()}")
    
    with col3:
        confidence = sentiment.get('confidence', 0)
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    
    # Sentiment details
    st.markdown("#### 📝 Analysis Details")
    
    analysis_text = f"""
    **Urgency Assessment:** The text shows {urgency} urgency indicators based on language patterns and keyword analysis.
    
    **Emotional Tone:** The overall emotional tone appears {emotion}, suggesting the complainant's state of mind during submission.
    
    **Confidence Level:** The sentiment analysis model is {confidence * 100:.1f}% confident in these assessments.
    """
    
    st.markdown(analysis_text)

def show_recommendations_analysis(results):
    """Display analysis recommendations"""
    
    recommendations = results.get('recommendations', [])
    
    if not recommendations:
        st.info("No specific recommendations were generated.")
        return
    
    st.markdown("#### 💡 Security Recommendations")
    
    # Categorize recommendations
    immediate_actions = []
    monitoring_actions = []
    general_actions = []
    
    for rec in recommendations:
        if any(word in rec.lower() for word in ['immediate', 'urgent', 'asap', 'escalate']):
            immediate_actions.append(rec)
        elif any(word in rec.lower() for word in ['monitor', 'watch', 'track', 'observe']):
            monitoring_actions.append(rec)
        else:
            general_actions.append(rec)
    
    # Display categorized recommendations
    if immediate_actions:
        st.markdown("##### 🚨 Immediate Actions Required")
        for action in immediate_actions:
            st.error(f"• {action}")
    
    if monitoring_actions:
        st.markdown("##### 👁️ Enhanced Monitoring")
        for action in monitoring_actions:
            st.warning(f"• {action}")
    
    if general_actions:
        st.markdown("##### 📋 General Recommendations")
        for action in general_actions:
            st.info(f"• {action}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Report"):
            st.info("Report generation functionality will be implemented.")
    
    with col2:
        if st.button("📧 Notify Team"):
            st.info("Team notification functionality will be implemented.")
    
    with col3:
        if st.button("📅 Create Tasks"):
            st.info("Task creation functionality will be implemented.")

def show_raw_analysis_data(results):
    """Display raw analysis data in JSON format"""
    
    st.markdown("#### 📊 Raw Analysis Data")
    st.markdown("Complete analysis results in JSON format for technical review:")
    
    # Format JSON for display
    formatted_json = json.dumps(results, indent=2, default=str)
    st.code(formatted_json, language='json')
    
    # Download button for raw data
    if st.button("💾 Download Raw Data"):
        st.info("Download functionality will be implemented.")

def show_recent_analysis_results(recent_cases):
    """Show recent analysis results overview"""
    
    if not recent_cases:
        st.info("No recent analysis results to display.")
        return
    
    # Summary table
    summary_data = []
    for case in recent_cases:
        results = case.get('analysis_results', {})
        summary_data.append({
            'Case ID': case['complaint_id'],
            'Classification': results.get('classification', 'Unknown').title(),
            'Threat Score': f"{results.get('threat_score', 0)}/10",
            'Entities Found': len(results.get('entities_found', [])),
            'Analyzed': datetime.fromisoformat(case.get('analysis_completed_at', case['timestamp'])).strftime('%Y-%m-%d %H:%M')
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

def show_analysis_progress():
    """Show analysis progress for running cases"""
    
    st.markdown("#### 🔄 Analysis in Progress")
    
    # Mock progress display
    progress_steps = [
        "✅ Evidence files processed",
        "🔄 Running NLP entity extraction...",
        "⏳ Threat classification pending",
        "⏳ Generating recommendations..."
    ]
    
    for step in progress_steps:
        if "✅" in step:
            st.success(step)
        elif "🔄" in step:
            st.warning(step)
        else:
            st.info(step)
    
    # Auto-refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
