"""
ForensiQ Timeline Analysis Engine
=================================
Advanced event timeline construction and temporal analysis for digital forensics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class TimelineEngine:
    """
    Advanced forensic timeline analysis engine with anomaly detection and correlation analysis
    """
    
    def __init__(self):
        self.events_df = None
        self.timeline_features = {}
        self.anomalies = []
        self.correlations = {}
        
    def load_data(self, data_path='data/processed'):
        """Load and parse forensic data for timeline analysis"""
        print("🕐 Loading forensic data for timeline analysis...")
        
        data_dir = Path(data_path)
        events = []
        
        # Load structured CSV data
        csv_files = list(data_dir.glob('*.csv'))
        for csv_file in csv_files:
            print(f"   📄 Processing: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                if 'Timestamp' in df.columns:
                    events.append(df)
                    print(f"   ✅ Loaded {len(df)} events from {csv_file.name}")
            except Exception as e:
                print(f"   ❌ Error loading {csv_file.name}: {e}")
        
        # Load text files with timestamp extraction
        txt_files = list(data_dir.glob('*.txt'))
        text_events = []
        
        for txt_file in txt_files[:10]:  # Sample first 10 files
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract timestamps using regex patterns
                timestamps = self._extract_timestamps(content)
                
                for timestamp in timestamps:
                    text_events.append({
                        'Timestamp': timestamp,
                        'Source_File': txt_file.name,
                        'Activity_Type': 'Log_Entry',
                        'Content_Preview': content[:200].replace('\n', ' ')
                    })
                    
            except Exception as e:
                print(f"   ❌ Error processing {txt_file.name}: {e}")
        
        if text_events:
            text_df = pd.DataFrame(text_events)
            events.append(text_df)
            print(f"   ✅ Extracted {len(text_events)} timestamps from text files")
        
        # Combine all events
        if events:
            self.events_df = pd.concat(events, ignore_index=True, sort=False)
            self._standardize_timestamps()
            print(f"📊 Total events loaded: {len(self.events_df)}")
            self._analyze_data_quality()
        else:
            print("❌ No valid event data found")
            
    def _extract_timestamps(self, text):
        """Extract timestamps from text using multiple patterns"""
        timestamp_patterns = [
            # Mon Jul 5 08:52:46.124 2004
            r'[A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?\s+\d{4}',
            # 2024-09-27 12:53:26.390859
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?',
            # 2024/09/27 12:53:26
            r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}',
            # Jul 5, 2004 08:52:46
            r'[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+\d{2}:\d{2}:\d{2}',
        ]
        
        timestamps = []
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, text)
            timestamps.extend(matches)
        
        return timestamps[:5]  # Limit to first 5 timestamps per file
    
    def _standardize_timestamps(self):
        """Standardize timestamp formats"""
        print("🔧 Standardizing timestamp formats...")
        
        def parse_timestamp(ts):
            """Try multiple timestamp formats"""
            if pd.isna(ts):
                return None
                
            ts_str = str(ts)
            
            # Try different formats
            formats = [
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%a %b %d %H:%M:%S.%f %Y',
                '%a %b %d %H:%M:%S %Y',
                '%b %d, %Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return pd.to_datetime(ts_str, format=fmt)
                except:
                    continue
                    
            # Fallback to pandas auto-parsing
            try:
                return pd.to_datetime(ts_str)
            except:
                return None
        
        # Apply timestamp parsing
        self.events_df['Timestamp'] = self.events_df['Timestamp'].apply(parse_timestamp)
        
        # Remove invalid timestamps
        valid_timestamps = self.events_df['Timestamp'].notna()
        print(f"   ✅ Valid timestamps: {valid_timestamps.sum()}/{len(self.events_df)}")
        
        self.events_df = self.events_df[valid_timestamps].copy()
        self.events_df = self.events_df.sort_values('Timestamp').reset_index(drop=True)
        
    def _analyze_data_quality(self):
        """Analyze timeline data quality"""
        print("\n📈 Timeline Data Quality Analysis:")
        print(f"   📊 Total Events: {len(self.events_df)}")
        
        if len(self.events_df) > 0:
            time_range = self.events_df['Timestamp'].max() - self.events_df['Timestamp'].min()
            print(f"   📅 Time Range: {time_range}")
            print(f"   🕐 Start: {self.events_df['Timestamp'].min()}")
            print(f"   🕐 End: {self.events_df['Timestamp'].max()}")
            
            # Activity distribution
            if 'Activity_Type' in self.events_df.columns:
                activity_counts = self.events_df['Activity_Type'].value_counts()
                print(f"   🎯 Activity Types: {len(activity_counts)}")
                for activity, count in activity_counts.head(5).items():
                    print(f"      - {activity}: {count} events")
    
    def build_timeline(self, time_unit='H', window_size=24):
        """Build comprehensive event timeline with temporal aggregation"""
        print(f"🔨 Building timeline with {time_unit} resolution...")
        
        if self.events_df is None or len(self.events_df) == 0:
            print("❌ No data available for timeline construction")
            return None
        
        # Create time-based features
        self.events_df['Hour'] = self.events_df['Timestamp'].dt.hour
        self.events_df['DayOfWeek'] = self.events_df['Timestamp'].dt.dayofweek
        self.events_df['Date'] = self.events_df['Timestamp'].dt.date
        
        # Temporal aggregation
        timeline = self.events_df.set_index('Timestamp').resample(time_unit).agg({
            'Activity_Type': 'count',
            'User_ID': lambda x: len(x.dropna().unique()) if 'User_ID' in x.name else 0,
            'IP_Address': lambda x: len(x.dropna().unique()) if 'IP_Address' in x.name else 0,
        }).rename(columns={'Activity_Type': 'Event_Count'})
        
        # Calculate temporal features
        timeline['Events_Per_Hour'] = timeline['Event_Count']
        timeline['Cumulative_Events'] = timeline['Event_Count'].cumsum()
        timeline['Moving_Avg_24h'] = timeline['Event_Count'].rolling(window=window_size, center=True).mean()
        
        self.timeline_features = {
            'total_events': len(self.events_df),
            'time_range_hours': (self.events_df['Timestamp'].max() - self.events_df['Timestamp'].min()).total_seconds() / 3600,
            'avg_events_per_hour': len(self.events_df) / max(1, (self.events_df['Timestamp'].max() - self.events_df['Timestamp'].min()).total_seconds() / 3600),
            'peak_hour': self.events_df['Hour'].mode().iloc[0] if len(self.events_df) > 0 else 0,
            'peak_day': self.events_df['DayOfWeek'].mode().iloc[0] if len(self.events_df) > 0 else 0
        }
        
        print(f"✅ Timeline built with {len(timeline)} time periods")
        return timeline
    
    def detect_temporal_anomalies(self, timeline_df, method='iqr', threshold=2.0):
        """Detect temporal anomalies in event patterns"""
        print(f"🔍 Detecting temporal anomalies using {method} method...")
        
        if timeline_df is None or len(timeline_df) == 0:
            return []
        
        anomalies = []
        event_counts = timeline_df['Event_Count'].dropna()
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = event_counts.quantile(0.25)
            Q3 = event_counts.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_mask = (event_counts < lower_bound) | (event_counts > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean_events = event_counts.mean()
            std_events = event_counts.std()
            z_scores = np.abs((event_counts - mean_events) / std_events)
            anomaly_mask = z_scores > threshold
            
        elif method == 'rolling_std':
            # Rolling standard deviation method
            rolling_mean = event_counts.rolling(window=24, center=True).mean()
            rolling_std = event_counts.rolling(window=24, center=True).std()
            anomaly_mask = np.abs(event_counts - rolling_mean) > (threshold * rolling_std)
        
        # Extract anomalous periods
        anomaly_indices = timeline_df.index[anomaly_mask]
        
        for idx in anomaly_indices:
            anomaly_info = {
                'timestamp': idx,
                'event_count': timeline_df.loc[idx, 'Event_Count'],
                'anomaly_type': 'High_Activity' if timeline_df.loc[idx, 'Event_Count'] > event_counts.median() else 'Low_Activity',
                'severity': 'High' if method == 'iqr' else 'Medium',
                'method_used': method
            }
            anomalies.append(anomaly_info)
        
        self.anomalies = anomalies
        print(f"   🚨 Found {len(anomalies)} temporal anomalies")
        
        return anomalies
    
    def correlate_events(self, correlation_window='1H'):
        """Analyze event correlations and patterns"""
        print(f"🔗 Analyzing event correlations within {correlation_window} windows...")
        
        if self.events_df is None or len(self.events_df) == 0:
            return {}
        
        correlations = {}
        
        # User activity correlations
        if 'User_ID' in self.events_df.columns and 'Activity_Type' in self.events_df.columns:
            user_activity = self.events_df.groupby(['User_ID', 'Activity_Type']).size().unstack(fill_value=0)
            if len(user_activity.columns) > 1:
                correlations['user_activity_correlation'] = user_activity.corr()
        
        # Temporal correlations
        hourly_activity = self.events_df.groupby(['Hour', 'Activity_Type']).size().unstack(fill_value=0)
        if len(hourly_activity.columns) > 1:
            correlations['hourly_activity_correlation'] = hourly_activity.corr()
        
        # IP address patterns
        if 'IP_Address' in self.events_df.columns:
            ip_activity = self.events_df.groupby(['IP_Address', 'Activity_Type']).size().unstack(fill_value=0)
            if len(ip_activity.columns) > 1 and len(ip_activity) > 1:
                correlations['ip_activity_correlation'] = ip_activity.corr()
        
        # Sequence analysis
        sequence_patterns = self._analyze_event_sequences()
        correlations['sequence_patterns'] = sequence_patterns
        
        self.correlations = correlations
        print(f"   ✅ Generated {len(correlations)} correlation analyses")
        
        return correlations
    
    def _analyze_event_sequences(self, max_sequences=10):
        """Analyze common event sequences"""
        if 'User_ID' not in self.events_df.columns or 'Activity_Type' not in self.events_df.columns:
            return {}
        
        sequences = {}
        
        # Group by user and analyze sequences
        for user_id in self.events_df['User_ID'].unique()[:20]:  # Limit to first 20 users
            user_events = self.events_df[self.events_df['User_ID'] == user_id].sort_values('Timestamp')
            
            if len(user_events) >= 2:
                # Create sequences of activities
                activities = user_events['Activity_Type'].tolist()
                
                # Generate bigrams (2-event sequences)
                bigrams = [(activities[i], activities[i+1]) for i in range(len(activities)-1)]
                
                for bigram in bigrams:
                    seq_key = f"{bigram[0]} → {bigram[1]}"
                    sequences[seq_key] = sequences.get(seq_key, 0) + 1
        
        # Return top sequences
        return dict(sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:max_sequences])
    
    def create_timeline_visualization(self, timeline_df, save_path='reports/screenshots'):
        """Create comprehensive timeline visualizations"""
        print("📊 Creating timeline visualizations...")
        
        if timeline_df is None or len(timeline_df) == 0:
            print("❌ No timeline data available for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=300)
        fig.suptitle('ForensiQ Timeline Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Event frequency over time
        axes[0, 0].plot(timeline_df.index, timeline_df['Event_Count'], color='#2E8B57', alpha=0.7, linewidth=2)
        if 'Moving_Avg_24h' in timeline_df.columns:
            axes[0, 0].plot(timeline_df.index, timeline_df['Moving_Avg_24h'], color='red', linewidth=2, label='24h Moving Average')
        
        # Highlight anomalies
        if self.anomalies:
            anomaly_times = [a['timestamp'] for a in self.anomalies]
            anomaly_counts = [timeline_df.loc[t, 'Event_Count'] for t in anomaly_times if t in timeline_df.index]
            axes[0, 0].scatter(anomaly_times, anomaly_counts, color='red', s=100, alpha=0.8, zorder=5, label='Anomalies')
        
        axes[0, 0].set_title('Event Frequency Timeline', fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Event Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Activity distribution by hour
        if len(self.events_df) > 0:
            hourly_dist = self.events_df['Hour'].value_counts().sort_index()
            axes[0, 1].bar(hourly_dist.index, hourly_dist.values, color='#4CAF50', alpha=0.7)
            axes[0, 1].set_title('Activity Distribution by Hour', fontweight='bold')
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Event Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative events over time
        if 'Cumulative_Events' in timeline_df.columns:
            axes[1, 0].plot(timeline_df.index, timeline_df['Cumulative_Events'], color='#FF6B35', linewidth=2)
            axes[1, 0].set_title('Cumulative Events Over Time', fontweight='bold')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Cumulative Count')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Activity type distribution
        if 'Activity_Type' in self.events_df.columns:
            activity_counts = self.events_df['Activity_Type'].value_counts().head(10)
            axes[1, 1].pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Top Activity Types', fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timeline_path = save_dir / 'timeline_analysis.png'
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Timeline visualization saved to: {timeline_path}")
        
        plt.close()
    
    def generate_timeline_report(self, save_path='reports'):
        """Generate comprehensive timeline analysis report"""
        print("📄 Generating timeline analysis report...")
        
        report = {
            'timeline_summary': {
                'total_events': len(self.events_df) if self.events_df is not None else 0,
                'time_range': {
                    'start': str(self.events_df['Timestamp'].min()) if self.events_df is not None and len(self.events_df) > 0 else None,
                    'end': str(self.events_df['Timestamp'].max()) if self.events_df is not None and len(self.events_df) > 0 else None,
                },
                'timeline_features': self.timeline_features
            },
            'anomalies': {
                'count': len(self.anomalies),
                'details': self.anomalies
            },
            'correlations_summary': {
                'analyses_performed': list(self.correlations.keys()) if self.correlations else [],
                'correlation_count': len(self.correlations)
            }
        }
        
        # Save report
        report_dir = Path(save_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / 'timeline_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✅ Timeline report saved to: {report_path}")
        return report


def main():
    """Main timeline analysis function"""
    print("🕐 ForensiQ Timeline Analysis Engine")
    print("=" * 50)
    
    # Initialize timeline engine
    engine = TimelineEngine()
    
    # Load data
    engine.load_data()
    
    if engine.events_df is not None and len(engine.events_df) > 0:
        # Build timeline
        timeline = engine.build_timeline(time_unit='H')
        
        # Detect anomalies
        anomalies = engine.detect_temporal_anomalies(timeline, method='iqr')
        
        # Analyze correlations
        correlations = engine.correlate_events()
        
        # Create visualizations
        engine.create_timeline_visualization(timeline)
        
        # Generate report
        report = engine.generate_timeline_report()
        
        print(f"\n🎉 Timeline Analysis Complete!")
        print(f"   📊 Events Analyzed: {len(engine.events_df)}")
        print(f"   🚨 Anomalies Found: {len(anomalies)}")
        print(f"   🔗 Correlations: {len(correlations)}")
        print("=" * 50)
        
    else:
        print("❌ No valid data available for timeline analysis")


if __name__ == "__main__":
    main()
