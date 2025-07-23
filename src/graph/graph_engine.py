"""
ForensiQ Graph Analysis Engine
==============================
Advanced actor-resource graph construction and network analysis for digital forensics
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class GraphEngine:
    """
    Advanced forensic graph analysis engine for actor-resource relationships and anomaly detection
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed multigraph for forensic relationships
        self.events_df = None
        self.graph_metrics = {}
        self.communities = {}
        self.anomalous_nodes = []
        self.anomalous_edges = []
        
    def load_data(self, data_path='data/processed'):
        """Load forensic data for graph construction"""
        print("🔗 Loading forensic data for graph analysis...")
        
        data_dir = Path(data_path)
        events = []
        
        # Load structured CSV data
        csv_files = list(data_dir.glob('*.csv'))
        for csv_file in csv_files:
            print(f"   📄 Processing: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                events.append(df)
                print(f"   ✅ Loaded {len(df)} events from {csv_file.name}")
            except Exception as e:
                print(f"   ❌ Error loading {csv_file.name}: {e}")
        
        if events:
            self.events_df = pd.concat(events, ignore_index=True, sort=False)
            print(f"📊 Total events loaded: {len(self.events_df)}")
            self._preprocess_data()
        else:
            print("❌ No valid event data found")
    
    def _preprocess_data(self):
        """Preprocess data for graph construction"""
        print("🔧 Preprocessing data for graph analysis...")
        
        # Clean and standardize data
        if 'User_ID' in self.events_df.columns:
            self.events_df['User_ID'] = self.events_df['User_ID'].astype(str)
        
        if 'IP_Address' in self.events_df.columns:
            self.events_df['IP_Address'] = self.events_df['IP_Address'].astype(str)
        
        if 'Resource_Accessed' in self.events_df.columns:
            self.events_df['Resource_Accessed'] = self.events_df['Resource_Accessed'].astype(str)
        
        # Create unified actor identifiers
        self.events_df['Actor'] = self.events_df.apply(self._create_actor_id, axis=1)
        
        print(f"   ✅ Preprocessed {len(self.events_df)} events")
        print(f"   👥 Unique actors: {self.events_df['Actor'].nunique()}")
        if 'Resource_Accessed' in self.events_df.columns:
            print(f"   📁 Unique resources: {self.events_df['Resource_Accessed'].nunique()}")
    
    def _create_actor_id(self, row):
        """Create unified actor identifier"""
        parts = []
        if 'User_ID' in row and pd.notna(row['User_ID']):
            parts.append(f"User_{row['User_ID']}")
        if 'IP_Address' in row and pd.notna(row['IP_Address']):
            parts.append(f"IP_{row['IP_Address']}")
        
        return "_".join(parts) if parts else "Unknown_Actor"
    
    def build_actor_resource_graph(self):
        """Build comprehensive actor-resource graph"""
        print("🏗️ Building actor-resource graph...")
        
        if self.events_df is None or len(self.events_df) == 0:
            print("❌ No data available for graph construction")
            return
        
        # Clear existing graph
        self.graph.clear()
        
        # Add actor nodes
        actors = self.events_df['Actor'].unique()
        for actor in actors:
            actor_data = self.events_df[self.events_df['Actor'] == actor]
            
            # Calculate actor attributes
            self.graph.add_node(actor, 
                              node_type='actor',
                              activity_count=len(actor_data),
                              unique_resources=actor_data['Resource_Accessed'].nunique() if 'Resource_Accessed' in actor_data.columns else 0,
                              activity_types=list(actor_data['Activity_Type'].unique()) if 'Activity_Type' in actor_data.columns else [],
                              risk_level=self._calculate_actor_risk(actor_data))
        
        # Add resource nodes
        if 'Resource_Accessed' in self.events_df.columns:
            resources = self.events_df['Resource_Accessed'].dropna().unique()
            for resource in resources:
                resource_data = self.events_df[self.events_df['Resource_Accessed'] == resource]
                
                self.graph.add_node(resource,
                                  node_type='resource',
                                  access_count=len(resource_data),
                                  unique_actors=resource_data['Actor'].nunique(),
                                  access_types=list(resource_data['Activity_Type'].unique()) if 'Activity_Type' in resource_data.columns else [],
                                  sensitivity=self._calculate_resource_sensitivity(resource))
        
        # Add edges (actor-resource relationships)
        for _, event in self.events_df.iterrows():
            if pd.notna(event.get('Resource_Accessed')):
                actor = event['Actor']
                resource = event['Resource_Accessed']
                activity = event.get('Activity_Type', 'Unknown')
                
                # Add edge with attributes
                self.graph.add_edge(actor, resource,
                                  activity_type=activity,
                                  timestamp=event.get('Timestamp'),
                                  action=event.get('Action', 'Unknown'),
                                  anomaly_type=event.get('Anomaly_Type'),
                                  label=event.get('Label', 'Normal'))
        
        print(f"   ✅ Graph constructed:")
        print(f"      📍 Nodes: {self.graph.number_of_nodes()}")
        print(f"      🔗 Edges: {self.graph.number_of_edges()}")
        print(f"      👥 Actors: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'actor'])}")
        print(f"      📁 Resources: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'resource'])}")
    
    def _calculate_actor_risk(self, actor_data):
        """Calculate risk level for an actor"""
        risk_score = 0
        
        # High activity count
        if len(actor_data) > actor_data.shape[0] * 0.1:  # Top 10% most active
            risk_score += 2
        
        # Suspicious activities
        if 'Label' in actor_data.columns:
            suspicious_ratio = (actor_data['Label'] == 'Suspicious').mean()
            if suspicious_ratio > 0.5:
                risk_score += 3
            elif suspicious_ratio > 0.2:
                risk_score += 1
        
        # Anomaly types
        if 'Anomaly_Type' in actor_data.columns:
            if actor_data['Anomaly_Type'].notna().any():
                risk_score += 2
        
        # Failed actions
        if 'Action' in actor_data.columns:
            failed_ratio = (actor_data['Action'] == 'Failed').mean()
            if failed_ratio > 0.3:
                risk_score += 1
        
        # Convert to categorical risk
        if risk_score >= 5:
            return 'High'
        elif risk_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_resource_sensitivity(self, resource_path):
        """Calculate sensitivity level for a resource"""
        sensitive_keywords = ['confidential', 'secret', 'admin', 'password', 'key', 'private', 'secure']
        
        resource_lower = str(resource_path).lower()
        sensitivity_score = sum(1 for keyword in sensitive_keywords if keyword in resource_lower)
        
        if sensitivity_score >= 2:
            return 'High'
        elif sensitivity_score >= 1:
            return 'Medium'
        else:
            return 'Low'
    
    def calculate_graph_metrics(self):
        """Calculate comprehensive graph metrics"""
        print("📊 Calculating graph metrics...")
        
        if self.graph.number_of_nodes() == 0:
            print("❌ No graph available for metrics calculation")
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['basic'] = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }
        
        # Centrality measures
        print("   🎯 Calculating centrality measures...")
        
        # Convert to undirected for some metrics
        undirected_graph = self.graph.to_undirected()
        
        if self.graph.number_of_nodes() > 1:
            # Degree centrality
            degree_centrality = nx.degree_centrality(undirected_graph)
            
            # Betweenness centrality (limited for performance)
            if self.graph.number_of_nodes() <= 1000:
                betweenness_centrality = nx.betweenness_centrality(undirected_graph, k=min(100, self.graph.number_of_nodes()))
            else:
                betweenness_centrality = {}
            
            # Closeness centrality (limited for performance)
            if self.graph.number_of_nodes() <= 500:
                closeness_centrality = nx.closeness_centrality(undirected_graph)
            else:
                closeness_centrality = {}
            
            metrics['centrality'] = {
                'degree': degree_centrality,
                'betweenness': betweenness_centrality,
                'closeness': closeness_centrality
            }
            
            # Top central nodes
            metrics['top_nodes'] = {
                'highest_degree': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
                'highest_betweenness': sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10] if betweenness_centrality else []
            }
        
        # Component analysis
        if nx.is_weakly_connected(self.graph):
            metrics['components'] = {'largest_component_size': self.graph.number_of_nodes()}
        else:
            components = list(nx.weakly_connected_components(self.graph))
            metrics['components'] = {
                'num_components': len(components),
                'largest_component_size': len(max(components, key=len)),
                'component_sizes': [len(comp) for comp in components]
            }
        
        # Actor vs Resource analysis
        actor_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'actor']
        resource_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'resource']
        
        metrics['node_types'] = {
            'actors': len(actor_nodes),
            'resources': len(resource_nodes),
            'actor_resource_ratio': len(actor_nodes) / max(1, len(resource_nodes))
        }
        
        # Risk distribution
        risk_distribution = {}
        for node, data in self.graph.nodes(data=True):
            risk = data.get('risk_level', 'Unknown')
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        metrics['risk_distribution'] = risk_distribution
        
        self.graph_metrics = metrics
        print(f"   ✅ Calculated {len(metrics)} metric categories")
        
        return metrics
    
    def detect_communities(self, method='louvain'):
        """Detect communities in the graph"""
        print(f"🔍 Detecting communities using {method} method...")
        
        if self.graph.number_of_nodes() <= 1:
            print("❌ Insufficient nodes for community detection")
            return {}
        
        # Convert to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        
        communities = {}
        
        if method == 'louvain' and undirected_graph.number_of_edges() > 0:
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(undirected_graph)
                
                # Group nodes by community
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)
                    
            except ImportError:
                print("   ⚠️ python-louvain not available, using connected components")
                components = nx.connected_components(undirected_graph)
                for i, component in enumerate(components):
                    communities[i] = list(component)
        
        elif method == 'connected_components':
            components = nx.connected_components(undirected_graph)
            for i, component in enumerate(components):
                communities[i] = list(component)
        
        # Analyze communities
        community_analysis = {}
        for comm_id, nodes in communities.items():
            actors = [n for n in nodes if self.graph.nodes[n].get('node_type') == 'actor']
            resources = [n for n in nodes if self.graph.nodes[n].get('node_type') == 'resource']
            
            community_analysis[comm_id] = {
                'size': len(nodes),
                'actors': len(actors),
                'resources': len(resources),
                'nodes': nodes[:10]  # Sample of nodes
            }
        
        self.communities = {
            'partition': communities,
            'analysis': community_analysis,
            'num_communities': len(communities)
        }
        
        print(f"   ✅ Found {len(communities)} communities")
        return self.communities
    
    def detect_graph_anomalies(self):
        """Detect anomalous nodes and edges in the graph"""
        print("🚨 Detecting graph anomalies...")
        
        if self.graph.number_of_nodes() == 0:
            return [], []
        
        anomalous_nodes = []
        anomalous_edges = []
        
        # Node anomalies
        for node, data in self.graph.nodes(data=True):
            anomaly_score = 0
            reasons = []
            
            # High degree nodes
            degree = self.graph.degree(node)
            if degree > np.percentile([self.graph.degree(n) for n in self.graph.nodes()], 95):
                anomaly_score += 2
                reasons.append("High degree connectivity")
            
            # High risk actors
            if data.get('risk_level') == 'High':
                anomaly_score += 3
                reasons.append("High risk level")
            
            # Highly accessed resources
            if data.get('node_type') == 'resource' and data.get('access_count', 0) > 50:
                anomaly_score += 2
                reasons.append("Highly accessed resource")
            
            # Sensitive resources with many accessors
            if (data.get('node_type') == 'resource' and 
                data.get('sensitivity') == 'High' and 
                data.get('unique_actors', 0) > 10):
                anomaly_score += 3
                reasons.append("Sensitive resource with many accessors")
            
            if anomaly_score >= 3:
                anomalous_nodes.append({
                    'node': node,
                    'score': anomaly_score,
                    'reasons': reasons,
                    'type': data.get('node_type', 'unknown')
                })
        
        # Edge anomalies
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            anomaly_score = 0
            reasons = []
            
            # Suspicious activity types
            if data.get('label') == 'Suspicious':
                anomaly_score += 3
                reasons.append("Suspicious activity")
            
            # Failed actions
            if data.get('action') == 'Failed':
                anomaly_score += 1
                reasons.append("Failed action")
            
            # Anomaly types
            if data.get('anomaly_type'):
                anomaly_score += 2
                reasons.append(f"Anomaly type: {data.get('anomaly_type')}")
            
            if anomaly_score >= 2:
                anomalous_edges.append({
                    'edge': (source, target),
                    'score': anomaly_score,
                    'reasons': reasons,
                    'activity': data.get('activity_type', 'unknown')
                })
        
        self.anomalous_nodes = anomalous_nodes
        self.anomalous_edges = anomalous_edges
        
        print(f"   🚨 Found {len(anomalous_nodes)} anomalous nodes")
        print(f"   🚨 Found {len(anomalous_edges)} anomalous edges")
        
        return anomalous_nodes, anomalous_edges
    
    def visualize_graph(self, layout='spring', node_size_attr='activity_count', save_path='reports/screenshots'):
        """Create comprehensive graph visualizations"""
        print("📊 Creating graph visualizations...")
        
        if self.graph.number_of_nodes() == 0:
            print("❌ No graph available for visualization")
            return
        
        # Limit graph size for visualization
        if self.graph.number_of_nodes() > 200:
            print(f"   ⚠️ Large graph ({self.graph.number_of_nodes()} nodes), sampling top 200 nodes")
            # Sample top nodes by degree
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:200]
            subgraph = self.graph.subgraph([node for node, _ in top_nodes])
        else:
            subgraph = self.graph
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
        fig.suptitle('ForensiQ Graph Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Main graph visualization
        ax1 = axes[0, 0]
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.random_layout(subgraph)
        
        # Node colors and sizes
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            data = subgraph.nodes[node]
            
            # Color by node type
            if data.get('node_type') == 'actor':
                if data.get('risk_level') == 'High':
                    node_colors.append('#FF4444')  # Red for high-risk actors
                elif data.get('risk_level') == 'Medium':
                    node_colors.append('#FFA500')  # Orange for medium-risk actors
                else:
                    node_colors.append('#4CAF50')  # Green for low-risk actors
            else:
                if data.get('sensitivity') == 'High':
                    node_colors.append('#8B0000')  # Dark red for sensitive resources
                else:
                    node_colors.append('#87CEEB')  # Light blue for normal resources
            
            # Size by activity/access count
            size_value = data.get(node_size_attr, 1)
            node_sizes.append(max(100, min(1000, size_value * 20)))
        
        # Draw graph
        nx.draw(subgraph, pos, ax=ax1,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=False,
                edge_color='gray',
                alpha=0.7,
                arrows=True)
        
        ax1.set_title('Actor-Resource Network', fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF4444', label='High-Risk Actor'),
            Patch(facecolor='#FFA500', label='Medium-Risk Actor'),
            Patch(facecolor='#4CAF50', label='Low-Risk Actor'),
            Patch(facecolor='#8B0000', label='Sensitive Resource'),
            Patch(facecolor='#87CEEB', label='Normal Resource')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. Degree distribution
        ax2 = axes[0, 1]
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        ax2.hist(degrees, bins=min(20, len(set(degrees))), alpha=0.7, color='#2E8B57')
        ax2.set_title('Degree Distribution', fontweight='bold')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk distribution
        ax3 = axes[1, 0]
        if self.graph_metrics.get('risk_distribution'):
            risk_data = self.graph_metrics['risk_distribution']
            ax3.pie(risk_data.values(), labels=risk_data.keys(), autopct='%1.1f%%', startangle=90)
            ax3.set_title('Risk Level Distribution', fontweight='bold')
        
        # 4. Community visualization (if available)
        ax4 = axes[1, 1]
        if hasattr(self, 'communities') and self.communities.get('analysis'):
            comm_sizes = [info['size'] for info in self.communities['analysis'].values()]
            comm_ids = list(self.communities['analysis'].keys())
            ax4.bar(range(len(comm_sizes)), comm_sizes, alpha=0.7, color='#4CAF50')
            ax4.set_title('Community Sizes', fontweight='bold')
            ax4.set_xlabel('Community ID')
            ax4.set_ylabel('Size')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Community detection\nnot performed', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Community Analysis', fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        graph_path = save_dir / 'graph_analysis.png'
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Graph visualization saved to: {graph_path}")
        
        plt.close()
    
    def generate_graph_report(self, save_path='reports'):
        """Generate comprehensive graph analysis report"""
        print("📄 Generating graph analysis report...")
        
        report = {
            'graph_summary': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'actors': len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'actor']),
                'resources': len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'resource'])
            },
            'metrics': self.graph_metrics,
            'communities': {
                'num_communities': self.communities.get('num_communities', 0),
                'community_analysis': self.communities.get('analysis', {})
            },
            'anomalies': {
                'anomalous_nodes': len(self.anomalous_nodes),
                'anomalous_edges': len(self.anomalous_edges),
                'node_details': self.anomalous_nodes[:10],  # Top 10
                'edge_details': self.anomalous_edges[:10]   # Top 10
            }
        }
        
        # Save report
        report_dir = Path(save_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / 'graph_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✅ Graph report saved to: {report_path}")
        return report


def main():
    """Main graph analysis function"""
    print("🔗 ForensiQ Graph Analysis Engine")
    print("=" * 50)
    
    # Initialize graph engine
    engine = GraphEngine()
    
    # Load data
    engine.load_data()
    
    if engine.events_df is not None and len(engine.events_df) > 0:
        # Build graph
        engine.build_actor_resource_graph()
        
        # Calculate metrics
        metrics = engine.calculate_graph_metrics()
        
        # Detect communities
        communities = engine.detect_communities()
        
        # Detect anomalies
        anomalous_nodes, anomalous_edges = engine.detect_graph_anomalies()
        
        # Create visualizations
        engine.visualize_graph()
        
        # Generate report
        report = engine.generate_graph_report()
        
        print(f"\n🎉 Graph Analysis Complete!")
        print(f"   📍 Nodes: {engine.graph.number_of_nodes()}")
        print(f"   🔗 Edges: {engine.graph.number_of_edges()}")
        print(f"   🏘️ Communities: {communities.get('num_communities', 0)}")
        print(f"   🚨 Anomalous Nodes: {len(anomalous_nodes)}")
        print(f"   🚨 Anomalous Edges: {len(anomalous_edges)}")
        print("=" * 50)
        
    else:
        print("❌ No valid data available for graph analysis")


if __name__ == "__main__":
    main()
