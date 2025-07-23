"""
Timeline Reconstruction Page - ForensiQ Officer Interface
=======================================================
Advanced timeline reconstruction and temporal analysis for digital forensics
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from pathlib import Path
import sys

# Add src path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def show_timeline_reconstruction_page():
    st.markdown('<div class="main-header"><h2>⏰ Timeline Reconstruction</h2><p>Advanced temporal analysis and event correlation</p></div>', unsafe_allow_html=True)
    
    # Check if a case is selected
    selected_case = st.session_state.get('selected_case')
    
    if selected_case:
        show_case_timeline_analysis(selected_case)
    else:
        show_timeline_dashboard()

def show_timeline_dashboard():
    """Show timeline analysis dashboard for all cases"""
    
    all_complaints = st.session_state.complaints_db
    
    if not all_complaints:
        st.info("No cases available for timeline analysis. Cases will appear here once users submit complaints.")
        return
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", len(all_complaints))
    
    with col2:
        cases_with_evidence = len([c for c in all_complaints if c['evidence_files']])
        st.metric("Cases with Evidence", cases_with_evidence)
    
    with col3:
        recent_cases = len([c for c in all_complaints if 
                          datetime.fromisoformat(c['timestamp']) > datetime.now() - timedelta(days=7)])
        st.metric("Recent Cases (7d)", recent_cases)
    
    with col4:
        analyzed_cases = len([c for c in all_complaints if c.get('analysis_status') == 'completed'])
        st.metric("Analyzed Cases", analyzed_cases)
    
    st.markdown("---")
    
    # Timeline overview
    show_global_timeline_overview(all_complaints)
    
    st.markdown("---")
    
    # Case selection for detailed timeline
    st.markdown("### 🎯 Select Case for Timeline Reconstruction")
    
    # Filter cases
    cases_with_data = [c for c in all_complaints if c['evidence_files'] or c.get('analysis_status') == 'completed']
    
    if not cases_with_data:
        st.warning("No cases with sufficient data for timeline reconstruction.")
        return
    
    case_options = [f"{c['complaint_id']} - {c['incident']['title']} ({len(c['evidence_files'])} files)" 
                   for c in cases_with_data]
    
    selected_option = st.selectbox("Choose a case for detailed timeline analysis:", case_options)
    
    if selected_option:
        selected_idx = case_options.index(selected_option)
        selected_case = cases_with_data[selected_idx]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Case:** {selected_case['complaint_id']}")
            st.markdown(f"**Title:** {selected_case['incident']['title']}")
            st.markdown(f"**Type:** {selected_case['incident']['type']}")
            st.markdown(f"**Evidence Files:** {len(selected_case['evidence_files'])}")
        
        with col2:
            if st.button("🔍 Analyze Timeline", type="primary", use_container_width=True):
                st.session_state.selected_case = selected_case
                show_case_timeline_analysis(selected_case)

def show_global_timeline_overview(complaints):
    """Show global timeline overview of all cases"""
    
    st.markdown("### 📊 Global Timeline Overview")
    
    # Create timeline data
    timeline_events = []
    
    for complaint in complaints:
        # Add submission event
        timeline_events.append({
            'timestamp': datetime.fromisoformat(complaint['timestamp']),
            'event_type': 'Case Submitted',
            'case_id': complaint['complaint_id'],
            'priority': complaint['priority_score'],
            'description': f"Case {complaint['complaint_id']}: {complaint['incident']['title']}"
        })
        
        # Add incident date if different from submission
        incident_date = complaint['incident'].get('incident_date')
        if incident_date:
            try:
                incident_dt = datetime.fromisoformat(incident_date) if isinstance(incident_date, str) else incident_date
                if incident_dt != datetime.fromisoformat(complaint['timestamp']):
                    timeline_events.append({
                        'timestamp': incident_dt,
                        'event_type': 'Incident Occurred',
                        'case_id': complaint['complaint_id'],
                        'priority': complaint['priority_score'],
                        'description': f"Incident: {complaint['incident']['title']}"
                    })
            except:
                pass
        
        # Add analysis completion if available
        if complaint.get('analysis_completed_at'):
            timeline_events.append({
                'timestamp': datetime.fromisoformat(complaint['analysis_completed_at']),
                'event_type': 'Analysis Completed',
                'case_id': complaint['complaint_id'],
                'priority': complaint['priority_score'],
                'description': f"Analysis completed for {complaint['complaint_id']}"
            })
    
    if timeline_events:
        # Create timeline visualization
        timeline_df = pd.DataFrame(timeline_events)
        timeline_df = timeline_df.sort_values('timestamp')
        
        # Interactive timeline chart
        fig = go.Figure()
        
        # Color map for event types
        color_map = {
            'Case Submitted': '#2196F3',
            'Incident Occurred': '#F44336',
            'Analysis Completed': '#4CAF50'
        }
        
        for event_type in timeline_df['event_type'].unique():
            event_data = timeline_df[timeline_df['event_type'] == event_type]
            
            fig.add_trace(go.Scatter(
                x=event_data['timestamp'],
                y=[event_type] * len(event_data),
                mode='markers',
                marker=dict(
                    size=12,
                    color=color_map.get(event_type, '#666666'),
                    line=dict(width=1, color='white')
                ),
                text=event_data['description'],
                hovertemplate='<b>%{text}</b><br>%{x}<br>Priority: %{customdata}<extra></extra>',
                customdata=event_data['priority'],
                name=event_type
            ))
        
        fig.update_layout(
            title="Global Case Timeline",
            xaxis_title="Time",
            yaxis_title="Event Type",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Events", len(timeline_events))
        
        with col2:
            date_range = (timeline_df['timestamp'].max() - timeline_df['timestamp'].min()).days
            st.metric("Timeline Span", f"{date_range} days")
        
        with col3:
            avg_priority = timeline_df['priority'].mean()
            st.metric("Avg Priority", f"{avg_priority:.1f}")

def show_case_timeline_analysis(case):
    """Show detailed timeline analysis for a specific case"""
    
    st.markdown(f"### 🔍 Timeline Analysis: {case['complaint_id']}")
    st.markdown(f"**{case['incident']['title']}**")
    
    # Case timeline overview
    show_case_timeline_overview(case)
    
    st.markdown("---")
    
    # Timeline reconstruction tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Event Timeline", "🔍 Pattern Analysis", "🚨 Anomaly Detection", "📊 Correlation Analysis"])
    
    with tab1:
        show_event_timeline(case)
    
    with tab2:
        show_pattern_analysis(case)
    
    with tab3:
        show_anomaly_detection(case)
    
    with tab4:
        show_correlation_analysis(case)

def show_case_timeline_overview(case):
    """Show overview of case timeline"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📅 Key Timestamps")
        incident_date = case['incident'].get('incident_date', 'Unknown')
        discovered_date = case['incident'].get('discovered_date', 'Unknown')
        submitted_date = datetime.fromisoformat(case['timestamp']).strftime('%Y-%m-%d %H:%M')
        
        st.markdown(f"**Incident Occurred:** {incident_date}")
        st.markdown(f"**Incident Discovered:** {discovered_date}")
        st.markdown(f"**Case Submitted:** {submitted_date}")
        
        if case.get('analysis_completed_at'):
            analysis_date = datetime.fromisoformat(case['analysis_completed_at']).strftime('%Y-%m-%d %H:%M')
            st.markdown(f"**Analysis Completed:** {analysis_date}")
    
    with col2:
        st.markdown("#### 📁 Evidence Timeline")
        evidence_count = len(case['evidence_files'])
        st.markdown(f"**Evidence Files:** {evidence_count}")
        
        if evidence_count > 0:
            # Show evidence file timeline
            for i, evidence in enumerate(case['evidence_files'][:3]):  # Show first 3
                st.markdown(f"• {evidence['filename']} ({evidence.get('type', 'Unknown')})")
            
            if evidence_count > 3:
                st.markdown(f"• ... and {evidence_count - 3} more files")
    
    with col3:
        st.markdown("#### 🎯 Analysis Status")
        status = case.get('analysis_status', 'pending')
        st.markdown(f"**Status:** {status.title()}")
        
        if status == 'completed' and case.get('analysis_results'):
            results = case['analysis_results']
            threat_score = results.get('threat_score', 0)
            classification = results.get('classification', 'unknown')
            
            st.markdown(f"**Threat Score:** {threat_score}/10")
            st.markdown(f"**Classification:** {classification.title()}")

def show_event_timeline(case):
    """Show detailed event timeline for the case"""
    
    st.markdown("#### 📋 Reconstructed Event Timeline")
    
    # Generate mock timeline events based on case data
    timeline_events = generate_case_timeline_events(case)
    
    if not timeline_events:
        st.info("No timeline events could be reconstructed for this case.")
        return
    
    # Create interactive timeline
    fig = go.Figure()
    
    # Group events by type
    event_types = list(set(event['type'] for event in timeline_events))
    colors = px.colors.qualitative.Set3[:len(event_types)]
    
    for i, event_type in enumerate(event_types):
        type_events = [e for e in timeline_events if e['type'] == event_type]
        
        timestamps = [e['timestamp'] for e in type_events]
        y_positions = [i] * len(type_events)
        descriptions = [e['description'] for e in type_events]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_positions,
            mode='markers+lines',
            marker=dict(size=12, color=colors[i]),
            line=dict(color=colors[i], width=2),
            text=descriptions,
            hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>',
            name=event_type
        ))
    
    fig.update_layout(
        title=f"Event Timeline - {case['complaint_id']}",
        xaxis_title="Time",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(event_types))),
            ticktext=event_types
        ),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Event details table
    st.markdown("#### 📊 Event Details")
    
    event_df = pd.DataFrame([
        {
            'Timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Type': event['type'],
            'Description': event['description'],
            'Source': event.get('source', 'System'),
            'Confidence': f"{event.get('confidence', 0.8):.0%}"
        }
        for event in sorted(timeline_events, key=lambda x: x['timestamp'])
    ])
    
    st.dataframe(event_df, use_container_width=True, hide_index=True)

def generate_case_timeline_events(case):
    """Generate timeline events for a case"""
    
    events = []
    base_time = datetime.fromisoformat(case['timestamp'])
    
    # Case submission event
    events.append({
        'timestamp': base_time,
        'type': 'Case Management',
        'description': f"Case {case['complaint_id']} submitted",
        'source': 'ForensiQ System',
        'confidence': 1.0
    })
    
    # Incident occurrence (if different from submission)
    incident_date = case['incident'].get('incident_date')
    if incident_date:
        try:
            incident_time = datetime.fromisoformat(incident_date) if isinstance(incident_date, str) else incident_date
            events.append({
                'timestamp': incident_time,
                'type': 'Security Incident',
                'description': f"Incident occurred: {case['incident']['type']}",
                'source': 'Incident Report',
                'confidence': 0.9
            })
        except:
            pass
    
    # Evidence collection events
    evidence_time = base_time + timedelta(minutes=30)
    for evidence in case['evidence_files']:
        events.append({
            'timestamp': evidence_time,
            'type': 'Evidence Collection',
            'description': f"Evidence collected: {evidence['filename']}",
            'source': 'Evidence System',
            'confidence': 0.95
        })
        evidence_time += timedelta(minutes=15)
    
    # Analysis events
    if case.get('analysis_status') == 'completed':
        analysis_time = base_time + timedelta(hours=2)
        events.append({
            'timestamp': analysis_time,
            'type': 'Analysis',
            'description': "Automated analysis initiated",
            'source': 'Analysis Engine',
            'confidence': 1.0
        })
        
        if case.get('analysis_completed_at'):
            completion_time = datetime.fromisoformat(case['analysis_completed_at'])
            events.append({
                'timestamp': completion_time,
                'type': 'Analysis',
                'description': "Analysis completed",
                'source': 'Analysis Engine',
                'confidence': 1.0
            })
    
    # Simulated forensic events based on incident type
    incident_type = case['incident']['type'].lower()
    
    if 'phishing' in incident_type:
        events.extend([
            {
                'timestamp': base_time - timedelta(hours=2),
                'type': 'Network Activity',
                'description': "Suspicious email received",
                'source': 'Email Server',
                'confidence': 0.8
            },
            {
                'timestamp': base_time - timedelta(hours=1),
                'type': 'User Activity',
                'description': "User clicked suspicious link",
                'source': 'Web Proxy',
                'confidence': 0.7
            }
        ])
    
    elif 'malware' in incident_type:
        events.extend([
            {
                'timestamp': base_time - timedelta(hours=4),
                'type': 'Network Activity',
                'description': "Suspicious file download detected",
                'source': 'Network Monitor',
                'confidence': 0.8
            },
            {
                'timestamp': base_time - timedelta(hours=2),
                'type': 'System Activity',
                'description': "Malware execution detected",
                'source': 'Endpoint Security',
                'confidence': 0.9
            }
        ])
    
    elif 'breach' in incident_type:
        events.extend([
            {
                'timestamp': base_time - timedelta(hours=6),
                'type': 'Access Control',
                'description': "Unauthorized access attempt",
                'source': 'Access Control System',
                'confidence': 0.8
            },
            {
                'timestamp': base_time - timedelta(hours=3),
                'type': 'Data Access',
                'description': "Sensitive data accessed",
                'source': 'Database Audit',
                'confidence': 0.9
            }
        ])
    
    return events

def show_pattern_analysis(case):
    """Show pattern analysis for the case"""
    
    st.markdown("#### 🔍 Temporal Pattern Analysis")
    
    # Generate pattern analysis based on case data
    patterns = analyze_temporal_patterns(case)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Activity Patterns")
        
        # Time-based pattern analysis
        if patterns.get('hourly_distribution'):
            hourly_data = patterns['hourly_distribution']
            
            fig = go.Figure(data=go.Bar(
                x=list(range(24)),
                y=[hourly_data.get(h, 0) for h in range(24)],
                marker_color='#3498db'
            ))
            
            fig.update_layout(
                title="Activity by Hour of Day",
                xaxis_title="Hour",
                yaxis_title="Activity Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Day of week patterns
        if patterns.get('daily_distribution'):
            daily_data = patterns['daily_distribution']
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig = go.Figure(data=go.Bar(
                x=days,
                y=[daily_data.get(i, 0) for i in range(7)],
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                title="Activity by Day of Week",
                xaxis_title="Day",
                yaxis_title="Activity Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 🎯 Pattern Insights")
        
        insights = patterns.get('insights', [])
        
        if insights:
            for insight in insights:
                if insight['type'] == 'warning':
                    st.warning(f"⚠️ {insight['message']}")
                elif insight['type'] == 'info':
                    st.info(f"ℹ️ {insight['message']}")
                else:
                    st.success(f"✅ {insight['message']}")
        else:
            st.info("No significant patterns detected in current data.")
        
        # Pattern statistics
        st.markdown("##### 📈 Pattern Statistics")
        
        stats = patterns.get('statistics', {})
        
        if stats:
            st.metric("Peak Activity Hour", stats.get('peak_hour', 'Unknown'))
            st.metric("Most Active Day", stats.get('peak_day', 'Unknown'))
            st.metric("Activity Variance", f"{stats.get('variance', 0):.2f}")

def analyze_temporal_patterns(case):
    """Analyze temporal patterns in case data"""
    
    patterns = {
        'hourly_distribution': {},
        'daily_distribution': {},
        'insights': [],
        'statistics': {}
    }
    
    # Analyze case submission time
    submit_time = datetime.fromisoformat(case['timestamp'])
    submit_hour = submit_time.hour
    submit_day = submit_time.weekday()
    
    # Mock hourly distribution
    patterns['hourly_distribution'][submit_hour] = patterns['hourly_distribution'].get(submit_hour, 0) + 1
    patterns['daily_distribution'][submit_day] = patterns['daily_distribution'].get(submit_day, 0) + 1
    
    # Add some realistic distribution
    # Business hours typically see more activity
    business_hours = [9, 10, 11, 14, 15, 16]
    for hour in business_hours:
        patterns['hourly_distribution'][hour] = patterns['hourly_distribution'].get(hour, 0) + np.random.randint(1, 5)
    
    # Weekdays typically see more activity
    for day in range(5):  # Mon-Fri
        patterns['daily_distribution'][day] = patterns['daily_distribution'].get(day, 0) + np.random.randint(2, 8)
    
    # Generate insights
    if submit_hour < 6 or submit_hour > 22:
        patterns['insights'].append({
            'type': 'warning',
            'message': f"Case submitted outside business hours ({submit_hour:02d}:00) - may indicate urgency"
        })
    
    if submit_day >= 5:  # Weekend
        patterns['insights'].append({
            'type': 'warning',
            'message': "Case submitted on weekend - may indicate urgent incident"
        })
    
    if case['priority_score'] >= 7:
        patterns['insights'].append({
            'type': 'warning',
            'message': "High priority case - expedited analysis recommended"
        })
    
    # Statistics
    if patterns['hourly_distribution']:
        peak_hour = max(patterns['hourly_distribution'], key=patterns['hourly_distribution'].get)
        patterns['statistics']['peak_hour'] = f"{peak_hour:02d}:00"
    
    if patterns['daily_distribution']:
        peak_day_idx = max(patterns['daily_distribution'], key=patterns['daily_distribution'].get)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns['statistics']['peak_day'] = days[peak_day_idx]
    
    hourly_values = list(patterns['hourly_distribution'].values())
    if hourly_values:
        patterns['statistics']['variance'] = np.var(hourly_values)
    
    return patterns

def show_anomaly_detection(case):
    """Show anomaly detection for the case"""
    
    st.markdown("#### 🚨 Temporal Anomaly Detection")
    
    # Generate anomaly analysis
    anomalies = detect_temporal_anomalies(case)
    
    if not anomalies:
        st.success("✅ No temporal anomalies detected in this case.")
        return
    
    # Anomaly summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_anomalies = len(anomalies)
        st.metric("Total Anomalies", total_anomalies)
    
    with col2:
        high_severity = len([a for a in anomalies if a.get('severity') == 'High'])
        st.metric("High Severity", high_severity)
    
    with col3:
        if anomalies:
            avg_confidence = sum(a.get('confidence', 0) for a in anomalies) / len(anomalies)
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
    
    # Anomaly visualization
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        
        # Timeline with anomalies
        fig = go.Figure()
        
        # Normal events
        normal_times = [datetime.fromisoformat(case['timestamp'])]
        fig.add_trace(go.Scatter(
            x=normal_times,
            y=[0] * len(normal_times),
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Normal Events'
        ))
        
        # Anomalous events
        if 'timestamp' in anomaly_df.columns:
            anomaly_times = pd.to_datetime(anomaly_df['timestamp'])
            severity_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
            
            for severity in anomaly_df['severity'].unique():
                severity_data = anomaly_df[anomaly_df['severity'] == severity]
                severity_times = pd.to_datetime(severity_data['timestamp'])
                
                fig.add_trace(go.Scatter(
                    x=severity_times,
                    y=[1] * len(severity_times),
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=severity_colors.get(severity, 'gray'),
                        symbol='x'
                    ),
                    text=severity_data['description'],
                    hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>',
                    name=f'{severity} Severity'
                ))
        
        fig.update_layout(
            title="Anomaly Timeline",
            xaxis_title="Time",
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Normal', 'Anomalous']
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details
        st.markdown("#### 📋 Anomaly Details")
        
        display_df = anomaly_df[['timestamp', 'type', 'severity', 'description', 'confidence']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0%}")
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

def detect_temporal_anomalies(case):
    """Detect temporal anomalies in case data"""
    
    anomalies = []
    submit_time = datetime.fromisoformat(case['timestamp'])
    
    # Check submission time anomalies
    if submit_time.hour < 6 or submit_time.hour > 22:
        anomalies.append({
            'timestamp': case['timestamp'],
            'type': 'Unusual Submission Time',
            'severity': 'Medium',
            'description': f"Case submitted at {submit_time.hour:02d}:00 (outside business hours)",
            'confidence': 0.8
        })
    
    if submit_time.weekday() >= 5:  # Weekend
        anomalies.append({
            'timestamp': case['timestamp'],
            'type': 'Weekend Submission',
            'severity': 'Low',
            'description': "Case submitted on weekend",
            'confidence': 0.6
        })
    
    # Check incident vs submission time gap
    incident_date = case['incident'].get('incident_date')
    if incident_date:
        try:
            incident_time = datetime.fromisoformat(incident_date)
            time_gap = submit_time - incident_time
            
            if time_gap.days > 30:
                anomalies.append({
                    'timestamp': case['timestamp'],
                    'type': 'Delayed Reporting',
                    'severity': 'High',
                    'description': f"Case reported {time_gap.days} days after incident",
                    'confidence': 0.9
                })
            elif time_gap.days > 7:
                anomalies.append({
                    'timestamp': case['timestamp'],
                    'type': 'Delayed Reporting',
                    'severity': 'Medium',
                    'description': f"Case reported {time_gap.days} days after incident",
                    'confidence': 0.7
                })
        except:
            pass
    
    # Check for rapid evidence collection
    if len(case['evidence_files']) > 5:
        anomalies.append({
            'timestamp': case['timestamp'],
            'type': 'High Evidence Volume',
            'severity': 'Medium',
            'description': f"Unusually high number of evidence files ({len(case['evidence_files'])})",
            'confidence': 0.7
        })
    
    # Check priority vs submission time correlation
    if case['priority_score'] >= 8 and (submit_time.hour >= 22 or submit_time.hour <= 6):
        anomalies.append({
            'timestamp': case['timestamp'],
            'type': 'High Priority After Hours',
            'severity': 'High',
            'description': "Critical incident reported outside business hours",
            'confidence': 0.85
        })
    
    return anomalies

def show_correlation_analysis(case):
    """Show correlation analysis for the case"""
    
    st.markdown("#### 📊 Event Correlation Analysis")
    
    # Generate correlation data
    correlations = perform_correlation_analysis(case)
    
    if not correlations:
        st.info("Insufficient data for correlation analysis.")
        return
    
    # Correlation metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correlation Pairs", correlations.get('total_pairs', 0))
    
    with col2:
        strong_corr = correlations.get('strong_correlations', 0)
        st.metric("Strong Correlations", strong_corr)
    
    with col3:
        if correlations.get('correlation_matrix') is not None:
            avg_corr = np.mean(np.abs(correlations['correlation_matrix']))
            st.metric("Avg Correlation", f"{avg_corr:.2f}")
    
    # Correlation matrix visualization
    if correlations.get('correlation_matrix') is not None:
        corr_matrix = correlations['correlation_matrix']
        labels = correlations.get('labels', [])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Event Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation insights
    if correlations.get('insights'):
        st.markdown("#### 💡 Correlation Insights")
        
        for insight in correlations['insights']:
            if insight['strength'] == 'Strong':
                st.error(f"🔴 **Strong Correlation:** {insight['description']}")
            elif insight['strength'] == 'Moderate':
                st.warning(f"🟡 **Moderate Correlation:** {insight['description']}")
            else:
                st.info(f"🔵 **Weak Correlation:** {insight['description']}")

def perform_correlation_analysis(case):
    """Perform correlation analysis on case data"""
    
    correlations = {}
    
    # Create feature vectors for correlation
    features = {
        'priority_score': case['priority_score'],
        'evidence_count': len(case['evidence_files']),
        'submission_hour': datetime.fromisoformat(case['timestamp']).hour,
        'submission_day': datetime.fromisoformat(case['timestamp']).weekday()
    }
    
    # Add incident type encoding
    incident_types = {
        'phishing attack': 1,
        'malware infection': 2,
        'data breach': 3,
        'ransomware attack': 4,
        'unauthorized access': 5,
        'other': 0
    }
    
    features['incident_type_code'] = incident_types.get(case['incident']['type'].lower(), 0)
    
    # Add analysis results if available
    if case.get('analysis_results'):
        results = case['analysis_results']
        features['threat_score'] = results.get('threat_score', 0)
        features['entities_count'] = len(results.get('entities_found', []))
    
    # Create correlation matrix
    feature_names = list(features.keys())
    feature_values = list(features.values())
    
    if len(feature_values) >= 3:
        # Create synthetic time series for correlation
        time_series_data = []
        for i, value in enumerate(feature_values):
            # Add some noise to create variation
            series = [value + np.random.normal(0, value * 0.1) for _ in range(10)]
            time_series_data.append(series)
        
        corr_matrix = np.corrcoef(time_series_data)
        
        correlations['correlation_matrix'] = corr_matrix
        correlations['labels'] = feature_names
        correlations['total_pairs'] = len(feature_names) * (len(feature_names) - 1) // 2
        
        # Find strong correlations
        strong_corr_count = 0
        insights = []
        
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_value = corr_matrix[i][j]
                
                if abs(corr_value) > 0.7:
                    strong_corr_count += 1
                    strength = 'Strong'
                elif abs(corr_value) > 0.5:
                    strength = 'Moderate'
                else:
                    strength = 'Weak'
                
                if abs(corr_value) > 0.5:  # Only report moderate+ correlations
                    insights.append({
                        'strength': strength,
                        'correlation': corr_value,
                        'description': f"{feature_names[i]} and {feature_names[j]} (r={corr_value:.2f})"
                    })
        
        correlations['strong_correlations'] = strong_corr_count
        correlations['insights'] = insights
    
    return correlations
