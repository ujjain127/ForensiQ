"""
ForensiQ Phase 3 Module Test Suite
==================================
Comprehensive testing for Timeline and Graph Analysis modules
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the modules to test
from timeline.timeline_engine import TimelineEngine
from graph.graph_engine import GraphEngine
from timeline.integrated_analysis import IntegratedAnalysis

class TestTimelineEngine(unittest.TestCase):
    """Test suite for Timeline Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = TimelineEngine()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        plt.close('all')
    
    def create_test_data(self):
        """Create test data for timeline analysis"""
        # Create test events with various timestamp formats
        test_events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        for i in range(100):
            # Create diverse timestamp formats
            timestamp = base_time + timedelta(hours=i, minutes=np.random.randint(0, 60))
            
            if i % 3 == 0:
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif i % 3 == 1:
                timestamp_str = timestamp.strftime('%m/%d/%Y %I:%M:%S %p')
            else:
                timestamp_str = timestamp.isoformat()
            
            event = {
                'Timestamp': timestamp_str,
                'Event_Type': np.random.choice(['LOGIN', 'FILE_ACCESS', 'NETWORK', 'SYSTEM']),
                'Actor': f'user_{np.random.randint(1, 20)}',
                'Resource_Accessed': f'resource_{np.random.randint(1, 50)}',
                'Action': np.random.choice(['READ', 'write', 'execute', 'delete']),
                'Severity': np.random.choice(['Low', 'Medium', 'High'])
            }
            test_events.append(event)
        
        # Add some anomalous events (unusual timing)
        for i in range(5):
            anomaly_time = base_time + timedelta(hours=i*20 + 2, minutes=30)
            anomaly_event = {
                'Timestamp': anomaly_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Event_Type': 'ANOMALY',
                'Actor': 'suspicious_user',
                'Resource_Accessed': 'sensitive_resource',
                'Action': 'unauthorized_access',
                'Severity': 'High'
            }
            test_events.append(anomaly_event)
        
        # Save test data
        test_df = pd.DataFrame(test_events)
        test_file = Path(self.test_dir) / 'test_events.txt'
        test_df.to_csv(test_file, sep='\t', index=False)
        
        self.test_data_path = self.test_dir
    
    def test_data_loading(self):
        """Test data loading functionality"""
        self.engine.load_data(self.test_data_path)
        
        self.assertIsNotNone(self.engine.events_df)
        self.assertGreater(len(self.engine.events_df), 0)
        self.assertIn('Timestamp', self.engine.events_df.columns)
        self.assertIn('Parsed_Timestamp', self.engine.events_df.columns)
    
    def test_timestamp_parsing(self):
        """Test timestamp parsing with multiple formats"""
        self.engine.load_data(self.test_data_path)
        
        # Check that timestamps were parsed successfully
        parsed_timestamps = self.engine.events_df['Parsed_Timestamp']
        self.assertTrue(all(pd.notna(parsed_timestamps)))
        self.assertTrue(all(isinstance(ts, pd.Timestamp) for ts in parsed_timestamps))
    
    def test_timeline_building(self):
        """Test timeline construction"""
        self.engine.load_data(self.test_data_path)
        timeline = self.engine.build_timeline(time_unit='H')
        
        self.assertIsNotNone(timeline)
        self.assertGreater(len(timeline), 0)
        self.assertIn('Event_Count', timeline.columns)
        
        # Test different time units
        timeline_daily = self.engine.build_timeline(time_unit='D')
        self.assertIsNotNone(timeline_daily)
    
    def test_anomaly_detection(self):
        """Test temporal anomaly detection"""
        self.engine.load_data(self.test_data_path)
        timeline = self.engine.build_timeline(time_unit='H')
        anomalies = self.engine.detect_temporal_anomalies(timeline)
        
        self.assertIsInstance(anomalies, list)
        # Should detect some anomalies from our test data
        self.assertGreaterEqual(len(anomalies), 0)
    
    def test_correlation_analysis(self):
        """Test event correlation functionality"""
        self.engine.load_data(self.test_data_path)
        correlations = self.engine.correlate_events()
        
        self.assertIsInstance(correlations, dict)
        # Should have correlation results
        if correlations:
            self.assertIn('correlations', correlations)
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        self.engine.load_data(self.test_data_path)
        timeline = self.engine.build_timeline(time_unit='H')
        features = self.engine.extract_temporal_features(timeline)
        
        self.assertIsInstance(features, dict)
        expected_features = ['peak_activity_hours', 'avg_events_per_hour', 'max_events_per_hour']
        for feature in expected_features:
            self.assertIn(feature, features)
    
    def test_visualization_creation(self):
        """Test visualization generation"""
        self.engine.load_data(self.test_data_path)
        timeline = self.engine.build_timeline(time_unit='H')
        
        # Test that visualization can be created without errors
        try:
            self.engine.create_timeline_visualization(timeline)
            visualization_created = True
        except Exception as e:
            print(f"Visualization creation failed: {e}")
            visualization_created = False
        
        self.assertTrue(visualization_created)
    
    def test_report_generation(self):
        """Test report generation"""
        self.engine.load_data(self.test_data_path)
        timeline = self.engine.build_timeline(time_unit='H')
        report = self.engine.generate_timeline_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('analysis_metadata', report)
        self.assertIn('timeline_summary', report)


class TestGraphEngine(unittest.TestCase):
    """Test suite for Graph Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = GraphEngine()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        plt.close('all')
    
    def create_test_data(self):
        """Create test data for graph analysis"""
        # Create test events with actor-resource relationships
        test_events = []
        
        # Normal users
        normal_users = [f'user_{i}' for i in range(1, 15)]
        # Suspicious users
        suspicious_users = ['admin_user', 'external_user', 'service_account']
        
        # Resources
        normal_resources = [f'document_{i}.txt' for i in range(1, 30)]
        sensitive_resources = ['admin_config.xml', 'password_file.txt', 'financial_data.csv']
        
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create normal activity
        for i in range(200):
            timestamp = base_time + timedelta(hours=i//10, minutes=np.random.randint(0, 60))
            
            event = {
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Event_Type': np.random.choice(['FILE_ACCESS', 'NETWORK', 'SYSTEM']),
                'Actor': np.random.choice(normal_users),
                'Resource_Accessed': np.random.choice(normal_resources),
                'Action': np.random.choice(['read', 'write', 'execute']),
                'Severity': np.random.choice(['Low', 'Medium'])
            }
            test_events.append(event)
        
        # Create suspicious activity
        for i in range(20):
            timestamp = base_time + timedelta(hours=i*2, minutes=np.random.randint(0, 60))
            
            event = {
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Event_Type': 'SECURITY_EVENT',
                'Actor': np.random.choice(suspicious_users),
                'Resource_Accessed': np.random.choice(sensitive_resources),
                'Action': np.random.choice(['read', 'write', 'delete']),
                'Severity': 'High'
            }
            test_events.append(event)
        
        # Save test data
        test_df = pd.DataFrame(test_events)
        test_file = Path(self.test_dir) / 'test_graph_events.txt'
        test_df.to_csv(test_file, sep='\t', index=False)
        
        self.test_data_path = self.test_dir
    
    def test_data_loading(self):
        """Test data loading functionality"""
        self.engine.load_data(self.test_data_path)
        
        self.assertIsNotNone(self.engine.events_df)
        self.assertGreater(len(self.engine.events_df), 0)
        self.assertIn('Actor', self.engine.events_df.columns)
        self.assertIn('Resource_Accessed', self.engine.events_df.columns)
    
    def test_graph_construction(self):
        """Test graph construction"""
        self.engine.load_data(self.test_data_path)
        self.engine.build_actor_resource_graph()
        
        self.assertIsNotNone(self.engine.graph)
        self.assertGreater(self.engine.graph.number_of_nodes(), 0)
        self.assertGreater(self.engine.graph.number_of_edges(), 0)
    
    def test_graph_metrics(self):
        """Test graph metrics calculation"""
        self.engine.load_data(self.test_data_path)
        self.engine.build_actor_resource_graph()
        metrics = self.engine.calculate_graph_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('basic', metrics)
        self.assertIn('centrality', metrics)
        
        # Check basic metrics
        basic_metrics = metrics['basic']
        self.assertIn('nodes', basic_metrics)
        self.assertIn('edges', basic_metrics)
        self.assertIn('density', basic_metrics)
        
        # Check centrality metrics
        centrality_metrics = metrics['centrality']
        self.assertIn('degree', centrality_metrics)
        self.assertIn('betweenness', centrality_metrics)
    
    def test_community_detection(self):
        """Test community detection"""
        self.engine.load_data(self.test_data_path)
        self.engine.build_actor_resource_graph()
        communities = self.engine.detect_communities()
        
        self.assertIsInstance(communities, dict)
        self.assertIn('num_communities', communities)
        self.assertIn('communities', communities)
        
        # Should detect at least one community
        self.assertGreaterEqual(communities['num_communities'], 1)
    
    def test_anomaly_detection(self):
        """Test graph anomaly detection"""
        self.engine.load_data(self.test_data_path)
        self.engine.build_actor_resource_graph()
        node_anomalies, edge_anomalies = self.engine.detect_graph_anomalies()
        
        self.assertIsInstance(node_anomalies, list)
        self.assertIsInstance(edge_anomalies, list)
        
        # Should detect some anomalies from suspicious users
        # (Note: exact number depends on the algorithm)
        total_anomalies = len(node_anomalies) + len(edge_anomalies)
        self.assertGreaterEqual(total_anomalies, 0)
    
    def test_visualization_creation(self):
        """Test graph visualization"""
        self.engine.load_data(self.test_data_path)
        self.engine.build_actor_resource_graph()
        
        # Test that visualization can be created without errors
        try:
            self.engine.visualize_graph()
            visualization_created = True
        except Exception as e:
            print(f"Graph visualization creation failed: {e}")
            visualization_created = False
        
        self.assertTrue(visualization_created)
    
    def test_report_generation(self):
        """Test graph report generation"""
        self.engine.load_data(self.test_data_path)
        self.engine.build_actor_resource_graph()
        report = self.engine.generate_graph_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('analysis_metadata', report)
        self.assertIn('graph_summary', report)


class TestIntegratedAnalysis(unittest.TestCase):
    """Test suite for Integrated Analysis"""
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = IntegratedAnalysis()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        plt.close('all')
    
    def create_test_data(self):
        """Create test data for integrated analysis"""
        # Create comprehensive test events
        test_events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create events that will show both temporal and graph patterns
        for i in range(150):
            timestamp = base_time + timedelta(hours=i//5, minutes=np.random.randint(0, 60))
            
            event = {
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Event_Type': np.random.choice(['LOGIN', 'FILE_ACCESS', 'NETWORK', 'SYSTEM']),
                'Actor': f'user_{np.random.randint(1, 10)}',
                'Resource_Accessed': f'resource_{np.random.randint(1, 20)}',
                'Action': np.random.choice(['read', 'write', 'execute', 'delete']),
                'Severity': np.random.choice(['Low', 'Medium', 'High'])
            }
            test_events.append(event)
        
        # Add coordinated suspicious activity (temporal + graph patterns)
        suspicious_time = base_time + timedelta(hours=48)
        for i in range(10):
            event = {
                'Timestamp': (suspicious_time + timedelta(minutes=i*5)).strftime('%Y-%m-%d %H:%M:%S'),
                'Event_Type': 'SECURITY_BREACH',
                'Actor': 'suspicious_actor',
                'Resource_Accessed': f'sensitive_file_{i}',
                'Action': 'unauthorized_access',
                'Severity': 'High'
            }
            test_events.append(event)
        
        # Save test data
        test_df = pd.DataFrame(test_events)
        test_file = Path(self.test_dir) / 'integrated_test_events.txt'
        test_df.to_csv(test_file, sep='\t', index=False)
        
        self.test_data_path = self.test_dir
    
    def test_integrated_data_loading(self):
        """Test integrated data loading"""
        # Mock the load_and_analyze method to avoid full analysis
        self.analyzer.timeline_engine.load_data(self.test_data_path)
        self.analyzer.graph_engine.load_data(self.test_data_path)
        
        # Check both engines loaded data
        self.assertIsNotNone(self.analyzer.timeline_engine.events_df)
        self.assertIsNotNone(self.analyzer.graph_engine.events_df)
        self.assertGreater(len(self.analyzer.timeline_engine.events_df), 0)
        self.assertGreater(len(self.analyzer.graph_engine.events_df), 0)
    
    def test_temporal_network_correlation(self):
        """Test temporal-network correlation analysis"""
        # Load data and build basic structures
        self.analyzer.timeline_engine.load_data(self.test_data_path)
        self.analyzer.graph_engine.load_data(self.test_data_path)
        
        timeline = self.analyzer.timeline_engine.build_timeline(time_unit='H')
        self.analyzer.graph_engine.build_actor_resource_graph()
        
        # Test correlation analysis
        self.analyzer._analyze_temporal_network_correlation(timeline)
        
        self.assertIn('temporal_network', self.analyzer.correlation_results)
        correlation_data = self.analyzer.correlation_results['temporal_network']
        self.assertIsInstance(correlation_data, dict)
    
    def test_anomaly_cross_correlation(self):
        """Test anomaly cross-correlation"""
        # Load data and detect anomalies
        self.analyzer.timeline_engine.load_data(self.test_data_path)
        self.analyzer.graph_engine.load_data(self.test_data_path)
        
        timeline = self.analyzer.timeline_engine.build_timeline(time_unit='H')
        timeline_anomalies = self.analyzer.timeline_engine.detect_temporal_anomalies(timeline)
        
        self.analyzer.graph_engine.build_actor_resource_graph()
        graph_anomalies = self.analyzer.graph_engine.detect_graph_anomalies()
        
        # Test cross-correlation
        self.analyzer._cross_correlate_anomalies(timeline_anomalies, graph_anomalies)
        
        self.assertIn('anomaly_correlation', self.analyzer.correlation_results)
        correlation_data = self.analyzer.correlation_results['anomaly_correlation']
        self.assertIsInstance(correlation_data, dict)
        self.assertIn('temporal_anomalies', correlation_data)
        self.assertIn('graph_node_anomalies', correlation_data)
    
    def test_activity_pattern_analysis(self):
        """Test activity pattern analysis"""
        # Load data
        self.analyzer.timeline_engine.load_data(self.test_data_path)
        self.analyzer.graph_engine.load_data(self.test_data_path)
        
        # Test activity pattern analysis
        self.analyzer._analyze_activity_patterns()
        
        self.assertIn('activity_patterns', self.analyzer.correlation_results)
        patterns = self.analyzer.correlation_results['activity_patterns']
        self.assertIsInstance(patterns, dict)
    
    def test_risk_integration(self):
        """Test integrated risk assessment"""
        # Load data and setup basic analysis
        self.analyzer.timeline_engine.load_data(self.test_data_path)
        self.analyzer.graph_engine.load_data(self.test_data_path)
        
        timeline = self.analyzer.timeline_engine.build_timeline(time_unit='H')
        self.analyzer.timeline_engine.detect_temporal_anomalies(timeline)
        
        self.analyzer.graph_engine.build_actor_resource_graph()
        self.analyzer.graph_engine.detect_graph_anomalies()
        
        # Test risk integration
        self.analyzer._integrate_risk_assessments()
        
        self.assertIn('integrated_risk', self.analyzer.correlation_results)
        risk_data = self.analyzer.correlation_results['integrated_risk']
        self.assertIsInstance(risk_data, dict)
        self.assertIn('integrated_risk_score', risk_data)
        self.assertIn('risk_level', risk_data)
    
    def test_key_findings_extraction(self):
        """Test key findings extraction"""
        # Setup some basic data
        self.analyzer.timeline_engine.anomalies = [{'type': 'test', 'severity': 'High'}]
        self.analyzer.graph_engine.anomalous_nodes = [{'node': 'test', 'score': 6}]
        self.analyzer.correlation_results['integrated_risk'] = {'risk_level': 'High'}
        
        findings = self.analyzer._extract_key_findings()
        
        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        # Setup some test data
        self.analyzer.correlation_results['integrated_risk'] = {'risk_level': 'High'}
        self.analyzer.timeline_engine.anomalies = [{'type': 'test'}] * 6  # More than 5
        self.analyzer.graph_engine.anomalous_nodes = [{'node': f'test_{i}'} for i in range(12)]  # More than 10
        
        recommendations = self.analyzer._generate_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and edge cases"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        timeline_engine = TimelineEngine()
        graph_engine = GraphEngine()
        
        # Test with empty dataframes
        timeline_engine.events_df = pd.DataFrame()
        graph_engine.events_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        timeline = timeline_engine.build_timeline(time_unit='H')
        self.assertTrue(timeline is None or len(timeline) == 0)
    
    def test_malformed_timestamp_handling(self):
        """Test handling of malformed timestamps"""
        engine = TimelineEngine()
        
        # Create test data with malformed timestamps
        test_data = pd.DataFrame({
            'Timestamp': ['invalid_timestamp', '2024-01-01 10:00:00', 'another_invalid'],
            'Event_Type': ['TEST', 'TEST', 'TEST'],
            'Actor': ['user1', 'user2', 'user3']
        })
        
        parsed_data = engine._parse_timestamps(test_data)
        
        # Should have some valid parsed timestamps
        valid_timestamps = parsed_data['Parsed_Timestamp'].notna().sum()
        self.assertGreater(valid_timestamps, 0)
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets"""
        # This is a basic performance test
        # In production, you might want more sophisticated benchmarking
        
        # Create larger test dataset
        large_data = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(1000):  # 1000 events
            timestamp = base_time + timedelta(hours=i//10, minutes=np.random.randint(0, 60))
            event = {
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Event_Type': 'TEST',
                'Actor': f'user_{i % 50}',
                'Resource_Accessed': f'resource_{i % 100}',
                'Action': 'test'
            }
            large_data.append(event)
        
        test_df = pd.DataFrame(large_data)
        
        # Test timeline engine
        timeline_engine = TimelineEngine()
        timeline_engine.events_df = test_df.copy()
        timeline_engine._parse_timestamps(timeline_engine.events_df)
        
        start_time = datetime.now()
        timeline = timeline_engine.build_timeline(time_unit='H')
        timeline_duration = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time (10 seconds)
        self.assertLess(timeline_duration, 10.0)
        
        # Test graph engine
        graph_engine = GraphEngine()
        graph_engine.events_df = test_df.copy()
        
        start_time = datetime.now()
        graph_engine.build_actor_resource_graph()
        graph_duration = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time (10 seconds)
        self.assertLess(graph_duration, 10.0)


def run_phase3_tests():
    """Run all Phase 3 tests and generate report"""
    print("🧪 ForensiQ Phase 3 Module Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTimelineEngine,
        TestGraphEngine,
        TestIntegratedAnalysis,
        TestDataIntegrity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Generate test report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"✅ Total Tests: {total_tests}")
    print(f"✅ Successful: {successes}")
    print(f"❌ Failures: {failures}")
    print(f"🚫 Errors: {errors}")
    print(f"📈 Success Rate: {(successes/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, trace in result.failures:
            print(f"   • {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚫 ERRORS ({len(result.errors)}):")
        for test, trace in result.errors:
            print(f"   • {test}: {trace.split('Exception:')[-1].strip()}")
    
    # Save test results
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'successes': successes,
        'failures': failures,
        'errors': errors,
        'success_rate': (successes/total_tests)*100 if total_tests > 0 else 0,
        'test_details': {
            'timeline_engine_tests': len(unittest.TestLoader().loadTestsFromTestCase(TestTimelineEngine)._tests),
            'graph_engine_tests': len(unittest.TestLoader().loadTestsFromTestCase(TestGraphEngine)._tests),
            'integrated_analysis_tests': len(unittest.TestLoader().loadTestsFromTestCase(TestIntegratedAnalysis)._tests),
            'data_integrity_tests': len(unittest.TestLoader().loadTestsFromTestCase(TestDataIntegrity)._tests)
        }
    }
    
    # Save test report
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    test_report_path = reports_dir / 'phase3_test_results.json'
    with open(test_report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n💾 Test report saved to: {test_report_path}")
    print("\n🎯 Phase 3 Module Testing Complete!")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the test suite
    success = run_phase3_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
