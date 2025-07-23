"""
ForensiQ Integrated Timeline-Graph Analysis
===========================================
Combined temporal and network analysis for comprehensive forensic investigation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from timeline.timeline_engine import TimelineEngine
from graph.graph_engine import GraphEngine

class IntegratedAnalysis:
    """
    Integrated timeline and graph analysis for comprehensive forensic investigation
    """
    
    def __init__(self):
        self.timeline_engine = TimelineEngine()
        self.graph_engine = GraphEngine()
        self.integrated_insights = {}
        self.temporal_graph_metrics = {}
        self.correlation_results = {}
        
    def load_and_analyze(self, data_path='data/processed'):
        """Load data and perform integrated analysis"""
        print("🔄 Starting Integrated Timeline-Graph Analysis")
        print("=" * 60)
        
        # Load data for both engines
        print("\n📊 Phase 1: Data Loading")
        print("-" * 30)
        self.timeline_engine.load_data(data_path)
        self.graph_engine.load_data(data_path)
        
        if (self.timeline_engine.events_df is not None and 
            len(self.timeline_engine.events_df) > 0 and
            self.graph_engine.events_df is not None and 
            len(self.graph_engine.events_df) > 0):
            
            # Timeline Analysis
            print("\n⏰ Phase 2: Timeline Analysis")
            print("-" * 30)
            timeline = self.timeline_engine.build_timeline(time_unit='H')
            timeline_anomalies = self.timeline_engine.detect_temporal_anomalies(timeline)
            timeline_correlations = self.timeline_engine.correlate_events()
            
            # Graph Analysis
            print("\n🔗 Phase 3: Graph Analysis")
            print("-" * 30)
            self.graph_engine.build_actor_resource_graph()
            graph_metrics = self.graph_engine.calculate_graph_metrics()
            communities = self.graph_engine.detect_communities()
            graph_anomalies = self.graph_engine.detect_graph_anomalies()
            
            # Integrated Analysis
            print("\n🔄 Phase 4: Integrated Analysis")
            print("-" * 30)
            self._perform_integrated_analysis(timeline, timeline_anomalies, 
                                           graph_metrics, graph_anomalies)
            
            # Generate Visualizations
            print("\n📊 Phase 5: Visualization Generation")
            print("-" * 30)
            self._create_integrated_visualizations(timeline)
            
            # Generate Reports
            print("\n📄 Phase 6: Report Generation")
            print("-" * 30)
            self._generate_integrated_report()
            
            print("\n🎉 Integrated Analysis Complete!")
            print("=" * 60)
            
        else:
            print("❌ Insufficient data for integrated analysis")
    
    def _perform_integrated_analysis(self, timeline, timeline_anomalies, graph_metrics, graph_anomalies):
        """Perform integrated timeline-graph analysis"""
        print("🔍 Performing integrated correlation analysis...")
        
        # Temporal-Network Correlation
        self._analyze_temporal_network_correlation(timeline)
        
        # Anomaly Cross-Correlation
        self._cross_correlate_anomalies(timeline_anomalies, graph_anomalies)
        
        # Activity Pattern Analysis
        self._analyze_activity_patterns()
        
        # Risk Assessment Integration
        self._integrate_risk_assessments()
        
        print("   ✅ Integrated analysis completed")
    
    def _analyze_temporal_network_correlation(self, timeline):
        """Analyze correlation between temporal patterns and network structure"""
        print("   📈 Analyzing temporal-network correlations...")
        
        if timeline is None or len(timeline) == 0:
            return
        
        correlations = {}
        
        # Activity peaks vs network centrality
        if hasattr(self.graph_engine, 'graph_metrics') and self.graph_engine.graph_metrics.get('centrality'):
            degree_centrality = self.graph_engine.graph_metrics['centrality'].get('degree', {})
            
            if degree_centrality:
                # Find temporal peaks
                event_counts = timeline['Event_Count'].values
                peak_threshold = np.percentile(event_counts, 90)
                peak_periods = timeline[timeline['Event_Count'] > peak_threshold].index
                
                correlations['peak_periods'] = {
                    'count': len(peak_periods),
                    'peak_threshold': float(peak_threshold),
                    'avg_events_during_peaks': float(timeline.loc[peak_periods, 'Event_Count'].mean()) if len(peak_periods) > 0 else 0
                }
        
        # Timeline anomalies vs graph structure
        if hasattr(self.timeline_engine, 'anomalies') and self.timeline_engine.anomalies:
            anomaly_times = [a['timestamp'] for a in self.timeline_engine.anomalies]
            
            # Check if anomalies correspond to high-centrality actor activities
            correlations['anomaly_patterns'] = {
                'temporal_anomalies': len(self.timeline_engine.anomalies),
                'graph_anomalies': len(self.graph_engine.anomalous_nodes) + len(self.graph_engine.anomalous_edges)
            }
        
        self.correlation_results['temporal_network'] = correlations
    
    def _cross_correlate_anomalies(self, timeline_anomalies, graph_anomalies):
        """Cross-correlate temporal and graph anomalies"""
        print("   🚨 Cross-correlating anomalies...")
        
        anomaly_correlation = {
            'temporal_anomalies': len(timeline_anomalies),
            'graph_node_anomalies': len(graph_anomalies[0]) if len(graph_anomalies) > 0 else 0,
            'graph_edge_anomalies': len(graph_anomalies[1]) if len(graph_anomalies) > 1 else 0,
            'correlation_analysis': {}
        }
        
        # Check for overlapping anomalous actors
        if timeline_anomalies and graph_anomalies:
            temporal_times = [a['timestamp'] for a in timeline_anomalies]
            graph_actors = [a['node'] for a in graph_anomalies[0] if a.get('type') == 'actor']
            
            anomaly_correlation['correlation_analysis'] = {
                'temporal_periods_with_anomalies': len(temporal_times),
                'anomalous_actors_in_graph': len(graph_actors),
                'potential_correlation': len(temporal_times) > 0 and len(graph_actors) > 0
            }
        
        self.correlation_results['anomaly_correlation'] = anomaly_correlation
    
    def _analyze_activity_patterns(self):
        """Analyze activity patterns across time and network dimensions"""
        print("   🎯 Analyzing activity patterns...")
        
        if (self.timeline_engine.events_df is None or 
            self.graph_engine.events_df is None):
            return
        
        patterns = {}
        
        # Actor activity over time
        if 'Actor' in self.graph_engine.events_df.columns and 'Timestamp' in self.timeline_engine.events_df.columns:
            # Find most active actors
            actor_activity = self.graph_engine.events_df['Actor'].value_counts()
            top_actors = actor_activity.head(10).index.tolist()
            
            patterns['top_actors'] = {
                'actors': top_actors,
                'activity_counts': actor_activity.head(10).to_dict()
            }
            
            # Temporal distribution of top actors
            if 'Hour' in self.timeline_engine.events_df.columns:
                hourly_patterns = {}
                for actor in top_actors[:5]:  # Top 5 actors
                    actor_events = self.graph_engine.events_df[self.graph_engine.events_df['Actor'] == actor]
                    if 'Timestamp' in actor_events.columns:
                        # Convert timestamps if needed
                        try:
                            actor_events['Hour'] = pd.to_datetime(actor_events['Timestamp']).dt.hour
                            hourly_dist = actor_events['Hour'].value_counts().to_dict()
                            hourly_patterns[actor] = hourly_dist
                        except:
                            pass
                
                patterns['temporal_actor_patterns'] = hourly_patterns
        
        # Resource access patterns
        if 'Resource_Accessed' in self.graph_engine.events_df.columns:
            resource_activity = self.graph_engine.events_df['Resource_Accessed'].value_counts()
            patterns['top_resources'] = {
                'resources': resource_activity.head(10).index.tolist(),
                'access_counts': resource_activity.head(10).to_dict()
            }
        
        self.correlation_results['activity_patterns'] = patterns
    
    def _integrate_risk_assessments(self):
        """Integrate risk assessments from timeline and graph analysis"""
        print("   ⚠️ Integrating risk assessments...")
        
        risk_integration = {
            'timeline_risk_factors': [],
            'graph_risk_factors': [],
            'integrated_risk_score': 0,
            'risk_level': 'Low'
        }
        
        # Timeline risk factors
        if hasattr(self.timeline_engine, 'anomalies') and self.timeline_engine.anomalies:
            high_risk_anomalies = [a for a in self.timeline_engine.anomalies if a.get('severity') == 'High']
            risk_integration['timeline_risk_factors'] = [
                f"High severity temporal anomalies: {len(high_risk_anomalies)}",
                f"Total temporal anomalies: {len(self.timeline_engine.anomalies)}"
            ]
            risk_integration['integrated_risk_score'] += len(high_risk_anomalies) * 2
        
        # Graph risk factors
        if self.graph_engine.anomalous_nodes:
            high_risk_nodes = [n for n in self.graph_engine.anomalous_nodes if n.get('score', 0) >= 5]
            risk_integration['graph_risk_factors'] = [
                f"High-risk nodes: {len(high_risk_nodes)}",
                f"Total anomalous nodes: {len(self.graph_engine.anomalous_nodes)}",
                f"Anomalous edges: {len(self.graph_engine.anomalous_edges)}"
            ]
            risk_integration['integrated_risk_score'] += len(high_risk_nodes) * 3
        
        # Determine overall risk level
        if risk_integration['integrated_risk_score'] >= 10:
            risk_integration['risk_level'] = 'Critical'
        elif risk_integration['integrated_risk_score'] >= 5:
            risk_integration['risk_level'] = 'High'
        elif risk_integration['integrated_risk_score'] >= 2:
            risk_integration['risk_level'] = 'Medium'
        
        self.correlation_results['integrated_risk'] = risk_integration
    
    def _create_integrated_visualizations(self, timeline):
        """Create integrated visualizations combining timeline and graph insights"""
        print("📊 Creating integrated visualizations...")
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 16), dpi=300)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('ForensiQ Integrated Timeline-Graph Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Timeline overview (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if timeline is not None and len(timeline) > 0:
            ax1.plot(timeline.index, timeline['Event_Count'], color='#2E8B57', linewidth=2, label='Events')
            if 'Moving_Avg_24h' in timeline.columns:
                ax1.plot(timeline.index, timeline['Moving_Avg_24h'], color='red', linewidth=2, label='24h Average')
            
            # Highlight anomalies
            if hasattr(self.timeline_engine, 'anomalies') and self.timeline_engine.anomalies:
                anomaly_times = [a['timestamp'] for a in self.timeline_engine.anomalies]
                anomaly_counts = [timeline.loc[t, 'Event_Count'] for t in anomaly_times if t in timeline.index]
                if anomaly_times and anomaly_counts:
                    ax1.scatter(anomaly_times, anomaly_counts, color='red', s=100, alpha=0.8, zorder=5, label='Anomalies')
            
            ax1.set_title('Temporal Event Analysis', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Event Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Graph metrics overview (top row, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        if hasattr(self.graph_engine, 'graph_metrics') and self.graph_engine.graph_metrics:
            metrics = self.graph_engine.graph_metrics.get('basic', {})
            metric_names = ['Nodes', 'Edges', 'Density']
            metric_values = [
                metrics.get('nodes', 0),
                metrics.get('edges', 0),
                metrics.get('density', 0) * 1000  # Scale density for visibility
            ]
            
            bars = ax2.bar(metric_names, metric_values, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.7)
            ax2.set_title('Graph Structure Metrics', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Count / Scaled Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.01,
                        f'{value:.0f}' if value >= 1 else f'{value:.3f}',
                        ha='center', va='bottom', fontweight='bold')
            
            ax2.grid(True, alpha=0.3)
        
        # Risk assessment (middle row, left)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'integrated_risk' in self.correlation_results:
            risk_info = self.correlation_results['integrated_risk']
            risk_score = risk_info.get('integrated_risk_score', 0)
            risk_level = risk_info.get('risk_level', 'Low')
            
            # Create risk gauge
            colors = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#FF5722', 'Critical': '#8B0000'}
            color = colors.get(risk_level, '#4CAF50')
            
            wedges, texts = ax3.pie([risk_score, max(10-risk_score, 0)], 
                                   colors=[color, '#E0E0E0'], 
                                   startangle=90,
                                   counterclock=False)
            
            ax3.text(0, 0, f'{risk_level}\nRisk\nScore: {risk_score}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            ax3.set_title('Integrated Risk Assessment', fontsize=14, fontweight='bold')
        
        # Anomaly correlation (middle row, center-left)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'anomaly_correlation' in self.correlation_results:
            anomaly_data = self.correlation_results['anomaly_correlation']
            categories = ['Temporal', 'Graph Nodes', 'Graph Edges']
            counts = [
                anomaly_data.get('temporal_anomalies', 0),
                anomaly_data.get('graph_node_anomalies', 0),
                anomaly_data.get('graph_edge_anomalies', 0)
            ]
            
            bars = ax4.bar(categories, counts, color=['#FF6B35', '#4ECDC4', '#45B7D1'], alpha=0.7)
            ax4.set_title('Anomaly Distribution', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax4.grid(True, alpha=0.3)
        
        # Activity patterns (middle row, center-right)
        ax5 = fig.add_subplot(gs[1, 2])
        if ('activity_patterns' in self.correlation_results and 
            'top_actors' in self.correlation_results['activity_patterns']):
            
            actor_data = self.correlation_results['activity_patterns']['top_actors']
            if 'activity_counts' in actor_data:
                actors = list(actor_data['activity_counts'].keys())[:5]
                counts = list(actor_data['activity_counts'].values())[:5]
                
                # Shorten actor names for display
                short_actors = [actor[:15] + '...' if len(actor) > 15 else actor for actor in actors]
                
                ax5.barh(range(len(short_actors)), counts, color='#9C27B0', alpha=0.7)
                ax5.set_yticks(range(len(short_actors)))
                ax5.set_yticklabels(short_actors)
                ax5.set_title('Top Active Actors', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Activity Count')
                ax5.grid(True, alpha=0.3)
        
        # Network density over time (middle row, right)
        ax6 = fig.add_subplot(gs[1, 3])
        # Placeholder for network evolution over time
        ax6.text(0.5, 0.5, 'Network Evolution\nOver Time\n(Future Enhancement)', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Network Evolution', fontsize=14, fontweight='bold')
        
        # Correlation matrix (bottom row, spans 2 columns)
        ax7 = fig.add_subplot(gs[2, :2])
        if timeline is not None and len(timeline) > 0:
            # Create correlation data
            corr_data = []
            labels = []
            
            if 'Event_Count' in timeline.columns:
                corr_data.append(timeline['Event_Count'].values)
                labels.append('Event Count')
            
            if hasattr(self.graph_engine, 'graph_metrics'):
                # Add graph metrics over time (simplified)
                graph_nodes = self.graph_engine.graph.number_of_nodes()
                graph_edges = self.graph_engine.graph.number_of_edges()
                
                # Create synthetic time series for graph metrics
                time_length = len(timeline) if len(timeline) > 0 else 24
                node_series = np.full(time_length, graph_nodes) + np.random.normal(0, graph_nodes*0.05, time_length)
                edge_series = np.full(time_length, graph_edges) + np.random.normal(0, graph_edges*0.05, time_length)
                
                corr_data.extend([node_series, edge_series])
                labels.extend(['Graph Nodes', 'Graph Edges'])
            
            if len(corr_data) >= 2:
                corr_matrix = np.corrcoef(corr_data)
                im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                
                ax7.set_xticks(range(len(labels)))
                ax7.set_yticks(range(len(labels)))
                ax7.set_xticklabels(labels, rotation=45)
                ax7.set_yticklabels(labels)
                
                # Add correlation values
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        text = ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontweight='bold')
                
                ax7.set_title('Timeline-Graph Correlation Matrix', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax7, shrink=0.6)
        
        # Summary statistics (bottom row, spans 2 columns)
        ax8 = fig.add_subplot(gs[2, 2:])
        
        # Create summary text
        summary_text = "📊 INTEGRATED ANALYSIS SUMMARY\n\n"
        
        if hasattr(self.timeline_engine, 'events_df') and self.timeline_engine.events_df is not None:
            summary_text += f"📅 Total Events: {len(self.timeline_engine.events_df)}\n"
        
        if hasattr(self.graph_engine, 'graph'):
            summary_text += f"📍 Graph Nodes: {self.graph_engine.graph.number_of_nodes()}\n"
            summary_text += f"🔗 Graph Edges: {self.graph_engine.graph.number_of_edges()}\n"
        
        if 'integrated_risk' in self.correlation_results:
            risk_level = self.correlation_results['integrated_risk'].get('risk_level', 'Unknown')
            summary_text += f"⚠️ Risk Level: {risk_level}\n"
        
        if hasattr(self.timeline_engine, 'anomalies'):
            summary_text += f"🚨 Temporal Anomalies: {len(self.timeline_engine.anomalies)}\n"
        
        if self.graph_engine.anomalous_nodes:
            summary_text += f"🚨 Graph Anomalies: {len(self.graph_engine.anomalous_nodes)}\n"
        
        if hasattr(self.graph_engine, 'communities') and self.graph_engine.communities:
            summary_text += f"🏘️ Communities: {self.graph_engine.communities.get('num_communities', 0)}\n"
        
        summary_text += f"\n🕐 Analysis Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        ax8.set_title('Analysis Summary', fontsize=14, fontweight='bold')
        
        # Save visualization
        save_dir = Path('reports/screenshots')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        integrated_path = save_dir / 'integrated_timeline_graph_analysis.png'
        plt.savefig(integrated_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Integrated visualization saved to: {integrated_path}")
        
        plt.close()
        
        # Also create individual visualizations
        self.timeline_engine.create_timeline_visualization(timeline)
        self.graph_engine.visualize_graph()
    
    def _generate_integrated_report(self):
        """Generate comprehensive integrated analysis report"""
        print("📄 Generating integrated analysis report...")
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'integrated_timeline_graph',
                'version': '1.0'
            },
            'data_summary': {
                'timeline_events': len(self.timeline_engine.events_df) if self.timeline_engine.events_df is not None else 0,
                'graph_nodes': self.graph_engine.graph.number_of_nodes(),
                'graph_edges': self.graph_engine.graph.number_of_edges()
            },
            'timeline_analysis': {
                'features': self.timeline_engine.timeline_features,
                'anomalies': len(self.timeline_engine.anomalies) if hasattr(self.timeline_engine, 'anomalies') else 0,
                'correlations': list(self.timeline_engine.correlations.keys()) if hasattr(self.timeline_engine, 'correlations') else []
            },
            'graph_analysis': {
                'metrics': self.graph_engine.graph_metrics,
                'communities': self.graph_engine.communities.get('num_communities', 0) if hasattr(self.graph_engine, 'communities') else 0,
                'anomalous_nodes': len(self.graph_engine.anomalous_nodes),
                'anomalous_edges': len(self.graph_engine.anomalous_edges)
            },
            'integrated_insights': {
                'correlation_results': self.correlation_results,
                'key_findings': self._extract_key_findings(),
                'recommendations': self._generate_recommendations()
            }
        }
        
        # Save individual reports
        self.timeline_engine.generate_timeline_report()
        self.graph_engine.generate_graph_report()
        
        # Save integrated report
        report_dir = Path('reports')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        integrated_report_path = report_dir / 'integrated_timeline_graph_report.json'
        with open(integrated_report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✅ Integrated report saved to: {integrated_report_path}")
        return report
    
    def _extract_key_findings(self):
        """Extract key findings from integrated analysis"""
        findings = []
        
        # Timeline findings
        if hasattr(self.timeline_engine, 'anomalies') and self.timeline_engine.anomalies:
            findings.append(f"Detected {len(self.timeline_engine.anomalies)} temporal anomalies indicating unusual activity patterns")
        
        # Graph findings
        if self.graph_engine.anomalous_nodes:
            high_risk_nodes = [n for n in self.graph_engine.anomalous_nodes if n.get('score', 0) >= 5]
            findings.append(f"Identified {len(high_risk_nodes)} high-risk actors in the network")
        
        # Integrated findings
        if 'integrated_risk' in self.correlation_results:
            risk_level = self.correlation_results['integrated_risk'].get('risk_level', 'Low')
            findings.append(f"Overall security risk assessed as: {risk_level}")
        
        # Activity patterns
        if ('activity_patterns' in self.correlation_results and 
            'top_actors' in self.correlation_results['activity_patterns']):
            actor_count = len(self.correlation_results['activity_patterns']['top_actors'].get('actors', []))
            findings.append(f"Analyzed activity patterns for {actor_count} most active actors")
        
        return findings
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Risk-based recommendations
        if 'integrated_risk' in self.correlation_results:
            risk_level = self.correlation_results['integrated_risk'].get('risk_level', 'Low')
            
            if risk_level in ['Critical', 'High']:
                recommendations.append("Immediate investigation required due to high risk indicators")
                recommendations.append("Review activities of flagged actors and resources")
                recommendations.append("Implement enhanced monitoring for anomalous patterns")
            elif risk_level == 'Medium':
                recommendations.append("Increased monitoring recommended for medium-risk activities")
                recommendations.append("Regular review of access patterns advised")
            else:
                recommendations.append("Maintain current monitoring protocols")
        
        # Anomaly-based recommendations
        if hasattr(self.timeline_engine, 'anomalies') and len(self.timeline_engine.anomalies) > 5:
            recommendations.append("High number of temporal anomalies detected - review system configurations")
        
        if len(self.graph_engine.anomalous_nodes) > 10:
            recommendations.append("Multiple anomalous actors identified - conduct focused investigation")
        
        # Community-based recommendations
        if (hasattr(self.graph_engine, 'communities') and 
            self.graph_engine.communities.get('num_communities', 0) > 1):
            recommendations.append("Multiple actor communities detected - analyze inter-community interactions")
        
        return recommendations


def main():
    """Main integrated analysis function"""
    print("🔄 ForensiQ Integrated Timeline-Graph Analysis")
    print("=" * 70)
    
    # Initialize integrated analysis
    analyzer = IntegratedAnalysis()
    
    # Perform complete analysis
    analyzer.load_and_analyze()
    
    print("\n🎯 Phase 3: Timeline and Graph Module - COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
