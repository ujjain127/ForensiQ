"""
Actor-Resource Graph Page - ForensiQ Officer Interface
=====================================================
Network analysis and actor-resource relationship visualization
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from pathlib import Path
import networkx as nx
import sys

# Add src path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def show_actor_graph_page():
    st.markdown('<div class="main-header"><h2>🕸️ Actor-Resource Graph</h2><p>Network analysis and relationship mapping</p></div>', unsafe_allow_html=True)
    
    # Check if a case is selected
    selected_case = st.session_state.get('selected_case')
    
    if selected_case:
        show_case_graph_analysis(selected_case)
    else:
        show_graph_dashboard()

def show_graph_dashboard():
    """Show graph analysis dashboard for all cases"""
    
    all_complaints = st.session_state.complaints_db
    
    if not all_complaints:
        st.info("No cases available for graph analysis. Cases will appear here once users submit complaints.")
        return
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_actors, total_resources, total_connections = calculate_global_graph_metrics(all_complaints)
    
    with col1:
        st.metric("Total Actors", total_actors)
    
    with col2:
        st.metric("Total Resources", total_resources)
    
    with col3:
        st.metric("Connections", total_connections)
    
    with col4:
        if total_actors > 0:
            density = total_connections / (total_actors * (total_actors - 1) / 2) if total_actors > 1 else 0
            st.metric("Network Density", f"{density:.3f}")
        else:
            st.metric("Network Density", "0.000")
    
    st.markdown("---")
    
    # Global network overview
    show_global_network_overview(all_complaints)
    
    st.markdown("---")
    
    # Case selection for detailed graph analysis
    st.markdown("### 🎯 Select Case for Graph Analysis")
    
    cases_with_analysis = [c for c in all_complaints if c.get('analysis_status') == 'completed' or c['evidence_files']]
    
    if not cases_with_analysis:
        st.warning("No cases with sufficient data for graph analysis.")
        return
    
    case_options = [f"{c['complaint_id']} - {c['incident']['title']} ({len(c['evidence_files'])} files)" 
                   for c in cases_with_analysis]
    
    selected_option = st.selectbox("Choose a case for detailed graph analysis:", case_options)
    
    if selected_option:
        selected_idx = case_options.index(selected_option)
        selected_case = cases_with_analysis[selected_idx]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Case:** {selected_case['complaint_id']}")
            st.markdown(f"**Title:** {selected_case['incident']['title']}")
            st.markdown(f"**Type:** {selected_case['incident']['type']}")
            if selected_case.get('analysis_results'):
                entities = len(selected_case['analysis_results'].get('entities_found', []))
                st.markdown(f"**Entities Found:** {entities}")
        
        with col2:
            if st.button("🔍 Analyze Graph", type="primary", use_container_width=True):
                st.session_state.selected_case = selected_case
                show_case_graph_analysis(selected_case)

def calculate_global_graph_metrics(complaints):
    """Calculate global graph metrics across all cases"""
    
    actors = set()
    resources = set()
    connections = 0
    
    for complaint in complaints:
        # Extract actors from submitter
        actors.add(complaint['submitter']['username'])
        
        # Extract entities from analysis if available
        if complaint.get('analysis_results', {}).get('entities_found'):
            for entity in complaint['analysis_results']['entities_found']:
                if entity['type'] in ['PERSON', 'ORGANIZATION']:
                    actors.add(entity['value'])
                elif entity['type'] in ['EMAIL', 'URL', 'IP_ADDRESS', 'PHONE']:
                    resources.add(entity['value'])
        
        # Add evidence files as resources
        for evidence in complaint['evidence_files']:
            resources.add(evidence['filename'])
        
        # Count connections (simplified)
        connections += len(complaint['evidence_files']) * 2  # Actor-Resource connections
    
    return len(actors), len(resources), connections

def show_global_network_overview(complaints):
    """Show global network overview visualization"""
    
    st.markdown("### 🌐 Global Network Overview")
    
    # Build global graph
    G = build_global_graph(complaints)
    
    if G.number_of_nodes() == 0:
        st.info("No network data available for visualization.")
        return
    
    # Network visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create network visualization
        show_network_visualization(G, "Global Actor-Resource Network")
    
    with col2:
        # Network statistics
        st.markdown("#### 📊 Network Statistics")
        
        actors = [n for n, d in G.nodes(data=True) if d.get('type') == 'actor']
        resources = [n for n, d in G.nodes(data=True) if d.get('type') == 'resource']
        
        st.metric("Actors", len(actors))
        st.metric("Resources", len(resources))
        st.metric("Edges", G.number_of_edges())
        
        if G.number_of_nodes() > 1:
            density = nx.density(G)
            st.metric("Density", f"{density:.3f}")
        
        # Top actors by degree
        if actors:
            degrees = dict(G.degree())
            top_actors = sorted(actors, key=lambda x: degrees.get(x, 0), reverse=True)[:3]
            
            st.markdown("#### 👑 Top Actors")
            for i, actor in enumerate(top_actors, 1):
                degree = degrees.get(actor, 0)
                st.markdown(f"{i}. {actor[:20]}... ({degree} connections)")

def build_global_graph(complaints):
    """Build a global graph from all complaints"""
    
    G = nx.Graph()
    
    for complaint in complaints:
        case_id = complaint['complaint_id']
        submitter = complaint['submitter']['username']
        
        # Add submitter as actor
        G.add_node(submitter, type='actor', label=submitter, case_ids=[case_id])
        
        # Add evidence files as resources
        for evidence in complaint['evidence_files']:
            resource_id = evidence['filename']
            G.add_node(resource_id, type='resource', label=resource_id, case_ids=[case_id])
            G.add_edge(submitter, resource_id, relationship='submitted_evidence', case_id=case_id)
        
        # Add entities from analysis
        if complaint.get('analysis_results', {}).get('entities_found'):
            for entity in complaint['analysis_results']['entities_found']:
                entity_id = entity['value']
                entity_type = 'actor' if entity['type'] in ['PERSON', 'ORGANIZATION'] else 'resource'
                
                if entity_id not in G:
                    G.add_node(entity_id, type=entity_type, label=entity_id, case_ids=[case_id])
                else:
                    # Update case_ids for existing node
                    if case_id not in G.nodes[entity_id].get('case_ids', []):
                        G.nodes[entity_id]['case_ids'].append(case_id)
                
                # Connect submitter to entity
                G.add_edge(submitter, entity_id, relationship='reported_entity', case_id=case_id)
    
    return G

def show_case_graph_analysis(case):
    """Show detailed graph analysis for a specific case"""
    
    st.markdown(f"### 🔍 Graph Analysis: {case['complaint_id']}")
    st.markdown(f"**{case['incident']['title']}**")
    
    # Build case-specific graph
    case_graph = build_case_graph(case)
    
    # Graph analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Network View", "📊 Graph Metrics", "🏘️ Communities", "🚨 Anomaly Detection"])
    
    with tab1:
        show_case_network_view(case, case_graph)
    
    with tab2:
        show_graph_metrics(case_graph)
    
    with tab3:
        show_community_analysis(case_graph)
    
    with tab4:
        show_graph_anomaly_detection(case, case_graph)

def build_case_graph(case):
    """Build graph for a specific case"""
    
    G = nx.Graph()
    case_id = case['complaint_id']
    submitter = case['submitter']['username']
    
    # Add submitter as central actor
    G.add_node(submitter, type='actor', label=submitter, role='submitter')
    
    # Add evidence files as resources
    for evidence in case['evidence_files']:
        resource_id = evidence['filename']
        G.add_node(resource_id, type='resource', label=resource_id, file_type=evidence.get('type', 'unknown'))
        G.add_edge(submitter, resource_id, relationship='submitted_evidence')
    
    # Add entities from analysis if available
    if case.get('analysis_results', {}).get('entities_found'):
        for entity in case['analysis_results']['entities_found']:
            entity_id = entity['value']
            entity_type = 'actor' if entity['type'] in ['PERSON', 'ORGANIZATION'] else 'resource'
            
            G.add_node(entity_id, type=entity_type, label=entity_id, 
                      entity_category=entity['type'], confidence=entity.get('confidence', 0.5))
            
            # Connect entities based on type
            if entity_type == 'actor':
                G.add_edge(submitter, entity_id, relationship='reported_actor')
            else:
                G.add_edge(submitter, entity_id, relationship='reported_resource')
    
    # Add synthetic relationships based on incident type
    add_synthetic_relationships(G, case)
    
    return G

def add_synthetic_relationships(G, case):
    """Add synthetic relationships based on incident patterns"""
    
    incident_type = case['incident']['type'].lower()
    
    # Get existing nodes by type
    actors = [n for n, d in G.nodes(data=True) if d.get('type') == 'actor']
    resources = [n for n, d in G.nodes(data=True) if d.get('type') == 'resource']
    
    if 'phishing' in incident_type:
        # Add email server and malicious links
        if not any('email' in r.lower() for r in resources):
            G.add_node('email_server', type='resource', label='Email Server', synthetic=True)
            for actor in actors[:2]:  # Connect to first 2 actors
                G.add_edge(actor, 'email_server', relationship='email_communication')
        
        if not any('malicious' in r.lower() for r in resources):
            G.add_node('malicious_link', type='resource', label='Malicious Link', synthetic=True)
            for actor in actors[:1]:  # Connect to first actor
                G.add_edge(actor, 'malicious_link', relationship='clicked_link')
    
    elif 'malware' in incident_type:
        # Add command & control server
        if not any('c2' in r.lower() or 'command' in r.lower() for r in resources):
            G.add_node('c2_server', type='resource', label='C&C Server', synthetic=True)
            for actor in actors[:1]:
                G.add_edge(actor, 'c2_server', relationship='malware_communication')
    
    elif 'breach' in incident_type:
        # Add database and unauthorized access points
        if not any('database' in r.lower() for r in resources):
            G.add_node('target_database', type='resource', label='Target Database', synthetic=True)
            for actor in actors[:2]:
                G.add_edge(actor, 'target_database', relationship='unauthorized_access')

def show_case_network_view(case, graph):
    """Show network visualization for the case"""
    
    st.markdown("#### 🕸️ Actor-Resource Network")
    
    if graph.number_of_nodes() == 0:
        st.info("No network data available for this case.")
        return
    
    # Network overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    actors = [n for n, d in graph.nodes(data=True) if d.get('type') == 'actor']
    resources = [n for n, d in graph.nodes(data=True) if d.get('type') == 'resource']
    
    with col1:
        st.metric("Actors", len(actors))
    
    with col2:
        st.metric("Resources", len(resources))
    
    with col3:
        st.metric("Connections", graph.number_of_edges())
    
    with col4:
        if graph.number_of_nodes() > 1:
            density = nx.density(graph)
            st.metric("Density", f"{density:.3f}")
    
    # Network visualization
    show_network_visualization(graph, f"Case {case['complaint_id']} Network")
    
    # Node and edge details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Actors")
        actor_data = []
        for actor in actors:
            degree = graph.degree(actor)
            role = graph.nodes[actor].get('role', 'unknown')
            actor_data.append({
                'Actor': actor[:30] + '...' if len(actor) > 30 else actor,
                'Role': role.title(),
                'Connections': degree,
                'Type': graph.nodes[actor].get('entity_category', 'User')
            })
        
        if actor_data:
            actor_df = pd.DataFrame(actor_data)
            st.dataframe(actor_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📁 Resources")
        resource_data = []
        for resource in resources[:10]:  # Show top 10
            degree = graph.degree(resource)
            res_type = graph.nodes[resource].get('file_type', graph.nodes[resource].get('entity_category', 'Unknown'))
            resource_data.append({
                'Resource': resource[:30] + '...' if len(resource) > 30 else resource,
                'Type': res_type.title(),
                'Connections': degree,
                'Synthetic': '✓' if graph.nodes[resource].get('synthetic') else ''
            })
        
        if resource_data:
            resource_df = pd.DataFrame(resource_data)
            st.dataframe(resource_df, use_container_width=True, hide_index=True)

def show_network_visualization(graph, title):
    """Create interactive network visualization"""
    
    if graph.number_of_nodes() == 0:
        st.info("No nodes to visualize.")
        return
    
    # Calculate layout
    try:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    except:
        pos = {node: (i % 5, i // 5) for i, node in enumerate(graph.nodes())}
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node attributes
        node_data = graph.nodes[node]
        node_type = node_data.get('type', 'unknown')
        degree = graph.degree(node)
        
        # Color by type
        if node_type == 'actor':
            color = '#FF6B6B'  # Red for actors
        else:
            color = '#4ECDC4'  # Teal for resources
        
        node_color.append(color)
        node_size.append(max(10, degree * 5))  # Size by degree
        
        # Hover text
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Type: {node_type.title()}<br>"
        hover_text += f"Connections: {degree}<br>"
        
        if node_data.get('confidence'):
            hover_text += f"Confidence: {node_data['confidence']:.0%}<br>"
        
        node_text.append(hover_text)
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='white')
        ),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Red: Actors, Teal: Resources",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_graph_metrics(graph):
    """Show detailed graph metrics"""
    
    st.markdown("#### 📊 Network Metrics")
    
    if graph.number_of_nodes() == 0:
        st.info("No graph data available for metrics calculation.")
        return
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📈 Basic Metrics")
        st.metric("Nodes", graph.number_of_nodes())
        st.metric("Edges", graph.number_of_edges())
        if graph.number_of_nodes() > 1:
            density = nx.density(graph)
            st.metric("Density", f"{density:.3f}")
    
    with col2:
        st.markdown("##### 🎯 Centrality")
        if graph.number_of_nodes() > 0:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(graph)
            top_degree_node = max(degree_centrality, key=degree_centrality.get)
            
            st.metric("Most Connected", top_degree_node[:20] + '...' if len(top_degree_node) > 20 else top_degree_node)
            st.metric("Centrality Score", f"{degree_centrality[top_degree_node]:.3f}")
            
            if nx.is_connected(graph):
                # Only for connected graphs
                try:
                    closeness_centrality = nx.closeness_centrality(graph)
                    top_closeness_node = max(closeness_centrality, key=closeness_centrality.get)
                    st.metric("Most Central", f"{closeness_centrality[top_closeness_node]:.3f}")
                except:
                    st.metric("Most Central", "N/A")
    
    with col3:
        st.markdown("##### 🌐 Structure")
        if graph.number_of_nodes() > 0:
            # Connected components
            num_components = nx.number_connected_components(graph)
            st.metric("Components", num_components)
            
            if num_components == 1:
                # Average path length for connected graphs
                try:
                    avg_path_length = nx.average_shortest_path_length(graph)
                    st.metric("Avg Path Length", f"{avg_path_length:.2f}")
                except:
                    st.metric("Avg Path Length", "N/A")
            
            # Clustering coefficient
            try:
                avg_clustering = nx.average_clustering(graph)
                st.metric("Clustering", f"{avg_clustering:.3f}")
            except:
                st.metric("Clustering", "N/A")
    
    # Degree distribution
    if graph.number_of_nodes() > 0:
        st.markdown("##### 📊 Degree Distribution")
        
        degrees = [graph.degree(n) for n in graph.nodes()]
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        
        fig = go.Figure(data=go.Bar(
            x=degree_counts.index,
            y=degree_counts.values,
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title="Node Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Number of Nodes",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_community_analysis(graph):
    """Show community detection analysis"""
    
    st.markdown("#### 🏘️ Community Analysis")
    
    if graph.number_of_nodes() < 3:
        st.info("Insufficient nodes for community detection (minimum 3 required).")
        return
    
    try:
        # Detect communities using Louvain method
        communities = nx.community.louvain_communities(graph)
        
        # Community metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Communities Found", len(communities))
        
        with col2:
            if communities:
                avg_size = sum(len(c) for c in communities) / len(communities)
                st.metric("Avg Community Size", f"{avg_size:.1f}")
        
        with col3:
            if communities:
                largest_community = max(len(c) for c in communities)
                st.metric("Largest Community", largest_community)
        
        # Community visualization
        if communities:
            st.markdown("##### 🎨 Community Visualization")
            
            # Assign colors to communities
            community_colors = px.colors.qualitative.Set3
            node_colors = {}
            
            for i, community in enumerate(communities):
                color = community_colors[i % len(community_colors)]
                for node in community:
                    node_colors[node] = color
            
            # Create community network plot
            try:
                pos = nx.spring_layout(graph, k=1, iterations=50)
            except:
                pos = {node: (i % 5, i // 5) for i, node in enumerate(graph.nodes())}
            
            # Prepare data
            node_x = []
            node_y = []
            node_text = []
            colors = []
            
            for node in graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Find community
                community_id = None
                for i, community in enumerate(communities):
                    if node in community:
                        community_id = i
                        break
                
                node_text.append(f"<b>{node}</b><br>Community: {community_id}")
                colors.append(node_colors.get(node, '#888888'))
            
            # Edge data
            edge_x = []
            edge_y = []
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create plot
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=15,
                    color=colors,
                    line=dict(width=1, color='white')
                ),
                showlegend=False
            ))
            
            fig.update_layout(
                title="Network Communities",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Community details
            st.markdown("##### 📋 Community Details")
            
            community_data = []
            for i, community in enumerate(communities):
                actors = [n for n in community if graph.nodes[n].get('type') == 'actor']
                resources = [n for n in community if graph.nodes[n].get('type') == 'resource']
                
                community_data.append({
                    'Community': i,
                    'Size': len(community),
                    'Actors': len(actors),
                    'Resources': len(resources),
                    'Members': ', '.join(list(community)[:3]) + ('...' if len(community) > 3 else '')
                })
            
            community_df = pd.DataFrame(community_data)
            st.dataframe(community_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Community detection failed: {str(e)}")

def show_graph_anomaly_detection(case, graph):
    """Show graph-based anomaly detection"""
    
    st.markdown("#### 🚨 Graph Anomaly Detection")
    
    if graph.number_of_nodes() == 0:
        st.info("No graph data available for anomaly detection.")
        return
    
    # Detect anomalies
    anomalies = detect_graph_anomalies(graph)
    
    if not anomalies:
        st.success("✅ No graph anomalies detected.")
        return
    
    # Anomaly summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Anomalies", len(anomalies))
    
    with col2:
        high_risk = len([a for a in anomalies if a.get('risk_level') == 'High'])
        st.metric("High Risk", high_risk)
    
    with col3:
        if anomalies:
            avg_score = sum(a.get('anomaly_score', 0) for a in anomalies) / len(anomalies)
            st.metric("Avg Anomaly Score", f"{avg_score:.2f}")
    
    # Anomaly visualization
    st.markdown("##### 🎯 Anomalous Nodes")
    
    anomaly_df = pd.DataFrame([
        {
            'Node': anomaly['node'][:30] + '...' if len(anomaly['node']) > 30 else anomaly['node'],
            'Type': anomaly['type'],
            'Risk Level': anomaly['risk_level'],
            'Score': f"{anomaly['anomaly_score']:.2f}",
            'Reason': anomaly['reason']
        }
        for anomaly in anomalies
    ])
    
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    
    # Anomaly insights
    st.markdown("##### 💡 Anomaly Insights")
    
    insights = generate_anomaly_insights(anomalies, case)
    
    for insight in insights:
        if insight['level'] == 'critical':
            st.error(f"🚨 **Critical:** {insight['message']}")
        elif insight['level'] == 'warning':
            st.warning(f"⚠️ **Warning:** {insight['message']}")
        else:
            st.info(f"ℹ️ **Info:** {insight['message']}")

def detect_graph_anomalies(graph):
    """Detect anomalies in the graph structure"""
    
    anomalies = []
    
    if graph.number_of_nodes() == 0:
        return anomalies
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(graph)
    degrees = [graph.degree(n) for n in graph.nodes()]
    
    if len(degrees) > 1:
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        
        # Detect nodes with unusually high degree (hubs)
        for node in graph.nodes():
            degree = graph.degree(node)
            
            if degree > mean_degree + 2 * std_degree:  # 2 standard deviations
                anomalies.append({
                    'node': node,
                    'type': 'Hub Anomaly',
                    'risk_level': 'High',
                    'anomaly_score': (degree - mean_degree) / max(std_degree, 1),
                    'reason': f'Unusually high connectivity ({degree} connections vs avg {mean_degree:.1f})'
                })
            
            # Detect isolated nodes
            elif degree == 0:
                anomalies.append({
                    'node': node,
                    'type': 'Isolation Anomaly',
                    'risk_level': 'Medium',
                    'anomaly_score': 2.0,
                    'reason': 'Isolated node with no connections'
                })
    
    # Detect nodes with unusual attributes
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        
        # Check for synthetic nodes (may indicate fabricated evidence)
        if data.get('synthetic'):
            anomalies.append({
                'node': node,
                'type': 'Synthetic Node',
                'risk_level': 'Medium',
                'anomaly_score': 1.5,
                'reason': 'Synthetically generated node based on incident patterns'
            })
        
        # Check for low confidence entities
        confidence = data.get('confidence', 1.0)
        if confidence < 0.5:
            anomalies.append({
                'node': node,
                'type': 'Low Confidence',
                'risk_level': 'Low',
                'anomaly_score': 1.0 - confidence,
                'reason': f'Entity extracted with low confidence ({confidence:.0%})'
            })
    
    return anomalies

def generate_anomaly_insights(anomalies, case):
    """Generate insights from detected anomalies"""
    
    insights = []
    
    # Hub anomalies
    hub_anomalies = [a for a in anomalies if a['type'] == 'Hub Anomaly']
    if hub_anomalies:
        insights.append({
            'level': 'critical',
            'message': f"{len(hub_anomalies)} nodes show unusually high connectivity - potential indicators of compromise"
        })
    
    # Isolation anomalies
    isolation_anomalies = [a for a in anomalies if a['type'] == 'Isolation Anomaly']
    if isolation_anomalies:
        insights.append({
            'level': 'warning',
            'message': f"{len(isolation_anomalies)} isolated nodes detected - may indicate incomplete data collection"
        })
    
    # Synthetic nodes
    synthetic_anomalies = [a for a in anomalies if a['type'] == 'Synthetic Node']
    if synthetic_anomalies:
        insights.append({
            'level': 'info',
            'message': f"{len(synthetic_anomalies)} synthetic relationships added based on incident type patterns"
        })
    
    # Low confidence entities
    low_conf_anomalies = [a for a in anomalies if a['type'] == 'Low Confidence']
    if low_conf_anomalies:
        insights.append({
            'level': 'warning',
            'message': f"{len(low_conf_anomalies)} entities have low extraction confidence - manual verification recommended"
        })
    
    # Case-specific insights
    if case['priority_score'] >= 7 and hub_anomalies:
        insights.append({
            'level': 'critical',
            'message': "High priority case with network anomalies - immediate investigation required"
        })
    
    return insights
