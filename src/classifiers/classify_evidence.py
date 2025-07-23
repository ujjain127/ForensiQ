"""
ForensiQ Evidence Classification Engine
======================================
Classifies digital evidence as Benign, Suspicious, or Malicious
Uses BERT embeddings + XGBoost classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import torch
import joblib
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceClassifier:
    """
    Main evidence classification engine combining BERT + XGBoost
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.bert_model = None
        self.xgb_classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Evidence classification labels
        self.labels = {
            0: 'BENIGN',
            1: 'SUSPICIOUS', 
            2: 'MALICIOUS'
        }
        
    def load_bert_model(self):
        """Load BERT tokenizer and model"""
        print("🤖 Loading BERT AI Model...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        logger.info(f"Loading BERT model: {self.model_name}")
        
        print("📥 Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print("🧠 Loading neural network weights...")
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        print("✅ BERT model loaded successfully!")
        
    def extract_bert_embeddings(self, texts, max_length=512):
        """
        Extract BERT embeddings for a list of texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length for BERT
            
        Returns:
            numpy array of embeddings (n_samples, 768)
        """
        if self.bert_model is None:
            self.load_bert_model()
            
        print(f"🔬 Extracting BERT embeddings for {len(texts)} texts...")
        print("   Converting text to high-dimensional vectors...")
        
        embeddings = []
        
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   Processing text {i+1}/{len(texts)} {'🔥' if i % 20 == 0 else '⚡'}")
            
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            ).to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        print(f"✅ Generated {len(embeddings)} embeddings of size {embeddings[0].shape[0]} dimensions!")
        return np.array(embeddings)
    
    def prepare_training_data(self, data_path='data/processed'):
        """
        Prepare training data from processed forensic files
        
        Returns:
            X: BERT embeddings
            y: Classification labels
        """
        print("🔍 Preparing forensic training data...")
        logger.info("Preparing training data...")
        
        # Load processed text files
        data_dir = Path(data_path)
        texts = []
        labels = []
        
        print(f"📂 Scanning directory: {data_path}")
        file_list = list(data_dir.glob('*.txt'))
        print(f"📄 Found {len(file_list)} forensic files to analyze")
        
        # Load and label data based on file patterns or content analysis
        for i, file_path in enumerate(file_list):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"   📖 Reading file {i+1}/{len(file_list)}: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Simple heuristic labeling (enhance with domain expertise)
            label = self._classify_content_heuristic(content, file_path.name)
            
            texts.append(content[:2000])  # Limit text length
            labels.append(label)
            
        print(f"✅ Loaded {len(texts)} forensic samples for training")
        
        # Show label distribution
        label_counts = {0: 0, 1: 0, 2: 0}
        for label in labels:
            label_counts[label] += 1
        
        print("📊 Label Distribution:")
        print(f"   🟢 BENIGN: {label_counts[0]} files")
        print(f"   🟡 SUSPICIOUS: {label_counts[1]} files") 
        print(f"   🔴 MALICIOUS: {label_counts[2]} files")
        
        # Extract BERT embeddings
        X = self.extract_bert_embeddings(texts)
        y = np.array(labels)
        
        return X, y
    
    def _classify_content_heuristic(self, content, filename):
        """
        Heuristic-based initial labeling for training data
        (Replace with expert-labeled data in production)
        """
        content_lower = content.lower()
        
        # Malicious indicators - Comprehensive forensic keyword set
        malicious_keywords = [
            # Malware & Threats
            'malware', 'virus', 'trojan', 'backdoor', 'exploit', 'rootkit',
            'keylogger', 'ransomware', 'spyware', 'adware', 'botnet', 'worm',
            'cryptocurrency miner', 'cryptojacker', 'rat', 'remote access trojan',
            'payload', 'shellcode', 'dropper', 'loader', 'stealer',
            
            # Attack Types
            'attack', 'breach', 'unauthorized', 'intrusion', 'infiltration',
            'compromise', 'hijack', 'injection', 'sql injection', 'xss',
            'buffer overflow', 'privilege escalation', 'lateral movement',
            'exfiltration', 'data theft', 'identity theft', 'phishing',
            'pharming', 'vishing', 'smishing', 'social engineering',
            
            # Hacking Activities
            'hack', 'hacker', 'cracking', 'password crack', 'brute force',
            'dictionary attack', 'credential stuffing', 'session hijacking',
            'man in the middle', 'mitm', 'eavesdropping', 'sniffing',
            'packet capture', 'wireshark', 'tcpdump', 'network scanning',
            
            # Cyber Weapons & Tools
            'metasploit', 'nmap', 'burp suite', 'kali linux', 'blackhat',
            'zero day', '0day', 'vulnerability scanner', 'penetration test',
            'reverse shell', 'bind shell', 'webshell', 'backdoor access',
            
            # Fraud & Financial Crime
            'credit card fraud', 'identity fraud', 'money laundering',
            'ponzi scheme', 'pyramid scheme', 'advance fee fraud',
            'romance scam', 'lottery scam', 'inheritance scam',
            'business email compromise', 'invoice fraud', 'wire fraud',
            
            # Dark Web & Illegal Activities
            'dark web', 'tor browser', 'onion site', 'illegal marketplace',
            'drug trafficking', 'weapon trafficking', 'child exploitation',
            'human trafficking', 'terrorist financing', 'money mule',
            
            # Data Breaches & Leaks
            'data breach', 'data leak', 'database dump', 'credential dump',
            'personal information exposed', 'pii leak', 'medical records leak',
            'financial data breach', 'customer data stolen',
            
            # Cryptocurrency & Blockchain Crimes
            'bitcoin theft', 'cryptocurrency scam', 'fake ico', 'pump and dump',
            'crypto mining malware', 'blockchain fraud', 'defi hack',
            'smart contract exploit', 'rug pull', 'exit scam'
        ]
        
        # Suspicious indicators - Enhanced forensic warning signs
        suspicious_keywords = [
            # System Anomalies
            'suspicious', 'unusual', 'anomaly', 'warning', 'alert', 'anomalous',
            'unexpected', 'irregular', 'abnormal', 'strange', 'odd behavior',
            'uncommon activity', 'rare event', 'baseline deviation',
            
            # Access & Authentication Issues
            'failed login', 'access denied', 'authentication failed',
            'invalid credentials', 'password mismatch', 'account locked',
            'multiple login attempts', 'login from unknown location',
            'unusual login time', 'concurrent sessions', 'session timeout',
            
            # Network & Connection Issues
            'connection refused', 'timeout', 'network error', 'dns error',
            'certificate error', 'ssl error', 'tls error', 'handshake failed',
            'port scan', 'network probe', 'unusual traffic', 'bandwidth spike',
            'connection spike', 'ddos', 'dos attack', 'flood attack',
            
            # File & System Issues
            'file corruption', 'missing file', 'modified file', 'deleted file',
            'unauthorized modification', 'permission denied', 'access violation',
            'memory violation', 'stack overflow', 'heap overflow',
            'segmentation fault', 'application crash', 'system instability',
            
            # User Behavior Anomalies
            'unusual user activity', 'after hours access', 'weekend access',
            'elevated privileges', 'admin rights requested', 'policy violation',
            'compliance violation', 'data access violation', 'file access violation',
            
            # Security Events
            'security event', 'security alert', 'security warning',
            'firewall blocked', 'antivirus detection', 'threat detected',
            'quarantine', 'isolated', 'blocked connection', 'denied request',
            
            # Performance & Resource Issues
            'high cpu usage', 'memory exhaustion', 'disk full', 'resource exhaustion',
            'performance degradation', 'slow response', 'service unavailable',
            'process spawn', 'unusual process', 'unknown process',
            
            # Communication Anomalies
            'encrypted communication', 'unknown protocol', 'suspicious email',
            'spam detected', 'phishing attempt', 'malicious link',
            'suspicious attachment', 'unknown sender', 'spoofed email',
            
            # Data Movement
            'large data transfer', 'unexpected download', 'unexpected upload',
            'data movement', 'file transfer', 'backup anomaly',
            'replication error', 'sync error', 'version mismatch',
            
            # Geographic & Time Anomalies
            'foreign ip', 'tor exit node', 'vpn detected', 'proxy detected',
            'geolocation mismatch', 'time zone anomaly', 'clock skew',
            'timestamp anomaly', 'log gap', 'missing logs'
        ]
        
        malicious_count = sum(1 for keyword in malicious_keywords if keyword in content_lower)
        suspicious_count = sum(1 for keyword in suspicious_keywords if keyword in content_lower)
        
        # Enhanced scoring with more granular detection
        if malicious_count >= 3:
            print(f"   🚨 HIGH THREAT MALICIOUS detected in {filename[:20]}... (red flags: {malicious_count})")
            return 2  # MALICIOUS
        elif malicious_count >= 2:
            print(f"   🚨 MALICIOUS detected in {filename[:20]}... (red flags: {malicious_count})")
            return 2  # MALICIOUS
        elif malicious_count >= 1 and suspicious_count >= 2:
            print(f"   ⚠️  ELEVATED SUSPICIOUS detected in {filename[:20]}... (mixed signals: M={malicious_count}, S={suspicious_count})")
            return 1  # SUSPICIOUS
        elif suspicious_count >= 3:
            print(f"   ⚠️  HIGH SUSPICIOUS detected in {filename[:20]}... (warnings: {suspicious_count})")
            return 1  # SUSPICIOUS
        elif suspicious_count >= 2 or malicious_count >= 1:
            if suspicious_count >= 2:
                print(f"   ⚠️  SUSPICIOUS detected in {filename[:20]}... (warnings: {suspicious_count})")
            return 1  # SUSPICIOUS
        else:
            return 0  # BENIGN
            
    def train_classifier(self, X, y, test_size=0.2):
        """
        Train XGBoost classifier on BERT embeddings
        """
        print("🚀 Training XGBoost classifier...")
        print(f"   📊 Training data shape: {X.shape}")
        print(f"   📋 Labels: {len(y)} samples")
        logger.info("Training XGBoost classifier...")
        
        # Split data
        print("✂️  Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"   🎯 Train set: {len(X_train)} samples")
        print(f"   🧪 Test set: {len(X_test)} samples")
        
        # Train XGBoost
        print("🌳 Building XGBoost decision trees...")
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        print("💪 Training in progress...")
        self.xgb_classifier.fit(X_train, y_train)
        print("✅ Training completed!")
        
        # Evaluate
        print("🔍 Evaluating model performance...")
        y_pred = self.xgb_classifier.predict(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📈 MODEL PERFORMANCE METRICS:")
        print("=" * 50)
        print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎪 Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"🎭 Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"🎨 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        # Store metrics for saving
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        logger.info("Classification Report:")
        print("\n📊 DETAILED CLASSIFICATION REPORT:")
        print("=" * 50)
        print(classification_report(y_test, y_pred, target_names=list(self.labels.values())))
        
        # Generate and save confusion matrix
        self.create_confusion_matrix(y_test, y_pred)
        
        return X_test, y_test, y_pred
    
    def predict(self, texts):
        """
        Predict classification for new texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predictions with confidence scores
        """
        if self.xgb_classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier() first.")
            
        # Extract embeddings
        X = self.extract_bert_embeddings(texts)
        
        # Predict
        predictions = self.xgb_classifier.predict(X)
        probabilities = self.xgb_classifier.predict_proba(X)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'label': self.labels[pred],
                'confidence': float(np.max(probabilities[i])),
                'probabilities': {
                    self.labels[j]: float(probabilities[i][j]) 
                    for j in range(len(self.labels))
                }
            })
            
        return results
    
    def create_confusion_matrix(self, y_test, y_pred):
        """
        Create and save confusion matrix visualization
        """
        print("📈 Creating confusion matrix visualization...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure with high DPI for quality
        plt.figure(figsize=(10, 8), dpi=300)
        
        # Create heatmap
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=list(self.labels.values()),
                   yticklabels=list(self.labels.values()),
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('ForensiQ Evidence Classification\nConfusion Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
        plt.ylabel('True Labels', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                   ha='center', fontsize=12, fontweight='bold')
        
        # Ensure directory exists
        reports_dir = Path('reports/screenshots')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        matrix_path = reports_dir / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Confusion matrix saved to: {matrix_path}")
        
        # Also save as PDF for presentations
        pdf_path = reports_dir / 'confusion_matrix.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"   📄 PDF version saved to: {pdf_path}")
        
        plt.close()
        
        return cm
    
    def create_metrics_visualization(self):
        """
        Create comprehensive metrics visualization
        """
        if not hasattr(self, 'metrics'):
            print("⚠️  No metrics available. Train the model first.")
            return
            
        print("📊 Creating performance metrics visualization...")
        
        # Create metrics bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        
        # Metrics bar chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [self.metrics['accuracy'], self.metrics['precision'], 
                         self.metrics['recall'], self.metrics['f1_score']]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}\n({value*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('ForensiQ Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Performance gauge
        ax2.pie([self.metrics['accuracy'], 1-self.metrics['accuracy']], 
               labels=['Correct', 'Incorrect'],
               colors=['#4CAF50', '#F44336'],
               startangle=90,
               counterclock=False,
               wedgeprops={'width': 0.3})
        
        ax2.set_title(f'Model Accuracy\n{self.metrics["accuracy"]*100:.1f}%', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        reports_dir = Path('reports/screenshots')
        metrics_path = reports_dir / 'performance_metrics.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Metrics visualization saved to: {metrics_path}")
        
        plt.close()
    
    def save_evaluation_report(self, X_test, y_test, y_pred):
        """
        Save comprehensive evaluation report as JSON
        """
        print("📋 Generating comprehensive evaluation report...")
        
        # Calculate per-class metrics
        report_dict = classification_report(y_test, y_pred, 
                                          target_names=list(self.labels.values()),
                                          output_dict=True)
        
        # Create comprehensive report
        evaluation_report = {
            'model_info': {
                'model_name': self.model_name,
                'training_samples': len(X_test) * 5,  # Approximate total samples
                'test_samples': len(X_test),
                'feature_dimensions': X_test.shape[1] if len(X_test) > 0 else 0
            },
            'overall_metrics': self.metrics,
            'per_class_metrics': report_dict,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Save as JSON
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / 'model_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        print(f"   💾 Evaluation report saved to: {report_path}")
        
        return evaluation_report
    
    def save_model(self, model_dir='models/classifier'):
        """Save trained models and evaluation results"""
        print("💾 Saving trained models and evaluation results...")
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost classifier
        print(f"   📁 Saving XGBoost classifier to {model_path}")
        joblib.dump(self.xgb_classifier, model_path / 'xgb_classifier.joblib')
        
        # Save metadata with metrics
        metadata = {
            'model_name': self.model_name,
            'labels': self.labels,
            'metrics': getattr(self, 'metrics', {}),
            'training_completed': True
        }
        joblib.dump(metadata, model_path / 'metadata.joblib')
        
        # Save metrics as JSON for easy reading
        if hasattr(self, 'metrics'):
            with open(model_path / 'metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"   📊 Performance metrics saved to {model_path}/metrics.json")
        
        print("✅ Models and evaluation results saved successfully!")
        logger.info(f"Models saved to {model_path}")
    
    def load_model(self, model_dir='models/classifier'):
        """Load pre-trained models"""
        model_path = Path(model_dir)
        
        # Load XGBoost classifier
        self.xgb_classifier = joblib.load(model_path / 'xgb_classifier.joblib')
        
        # Load metadata
        metadata = joblib.load(model_path / 'metadata.joblib')
        self.labels = metadata['labels']
        
        # Load BERT model
        self.load_bert_model()
        
        logger.info(f"Models loaded from {model_path}")

def main():
    """
    Main training pipeline for evidence classification
    """
    print("🎯 ForensiQ Evidence Classification Engine")
    print("=" * 50)
    print("🔬 Initializing BERT + XGBoost ML Pipeline...")
    
    # Initialize classifier
    classifier = EvidenceClassifier()
    
    # Prepare training data
    print("\n📚 PHASE 1: Data Preparation")
    print("-" * 30)
    X, y = classifier.prepare_training_data()
    
    # Train classifier
    print("\n🎓 PHASE 2: Model Training")
    print("-" * 30)
    X_test, y_test, y_pred = classifier.train_classifier(X, y)
    
    # Generate comprehensive evaluation
    print("\n📊 PHASE 3: Model Evaluation & Visualization")
    print("-" * 30)
    classifier.create_metrics_visualization()
    evaluation_report = classifier.save_evaluation_report(X_test, y_test, y_pred)
    
    # Save model
    print("\n💾 PHASE 4: Model Persistence")
    print("-" * 30)
    classifier.save_model()
    
    # Example prediction
    print("\n🔮 PHASE 5: Live Prediction Demo")
    print("-" * 30)
    sample_texts = [
        "Normal system startup completed successfully",
        "Multiple failed login attempts detected from unknown IP",
        "Malware signature detected in downloaded file"
    ]
    
    print("🧠 Running live predictions on sample texts...")
    predictions = classifier.predict(sample_texts)
    
    print("\n🔍 Sample Predictions:")
    for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
        risk_emoji = "🔴" if pred['label'] == 'MALICIOUS' else "🟡" if pred['label'] == 'SUSPICIOUS' else "🟢"
        print(f"{i+1}. {risk_emoji} Text: {text[:50]}...")
        print(f"   Label: {pred['label']} (Confidence: {pred['confidence']:.3f})")
        print()
    
    # Final summary
    print("🎉 ForensiQ Evidence Classification Complete!")
    print("=" * 50)
    if hasattr(classifier, 'metrics'):
        print(f"🎯 Final Model Accuracy: {classifier.metrics['accuracy']*100:.2f}%")
        print(f"📊 Evaluation reports saved to: reports/")
        print(f"📈 Visualizations saved to: reports/screenshots/")
    print("Ready for forensic investigation! 🕵️‍♂️")

if __name__ == "__main__":
    main()
