"""
ForensiQ Enhanced Evidence Classification Engine
==============================================
Advanced multi-model ensemble for 85%+ accuracy
Combines BERT embeddings, TF-IDF features, and statistical analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import torch
import joblib
import logging
from pathlib import Path
import json
import re
from collections import Counter
import textstat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEvidenceClassifier:
    """
    Enhanced evidence classification engine with multi-feature ensemble
    Target accuracy: 85%+
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.bert_model = None
        self.ensemble_classifier = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Evidence classification labels
        self.labels = {
            0: 'BENIGN',
            1: 'SUSPICIOUS', 
            2: 'MALICIOUS'
        }
        
        # Advanced forensic keyword patterns
        self.forensic_patterns = self._compile_forensic_patterns()
        
    def _compile_forensic_patterns(self):
        """Compile advanced forensic detection patterns"""
        patterns = {
            'malware_signatures': [
                r'\b(virus|trojan|backdoor|rootkit|keylogger|ransomware|spyware|botnet)\b',
                r'\b(payload|shellcode|dropper|loader|stealer|rat|cryptojacker)\b',
                r'\b(malware|exploit|zero[_\s-]?day|0day|vulnerability)\b'
            ],
            'attack_patterns': [
                r'\b(sql[_\s]?injection|xss|csrf|buffer[_\s]?overflow)\b',
                r'\b(privilege[_\s]?escalation|lateral[_\s]?movement|exfiltration)\b',
                r'\b(brute[_\s]?force|dictionary[_\s]?attack|credential[_\s]?stuffing)\b'
            ],
            'network_threats': [
                r'\b(ddos|dos[_\s]?attack|man[_\s]?in[_\s]?the[_\s]?middle|mitm)\b',
                r'\b(packet[_\s]?capture|network[_\s]?scanning|port[_\s]?scan)\b',
                r'\b(eavesdropping|sniffing|wireshark|tcpdump)\b'
            ],
            'suspicious_activities': [
                r'\b(failed[_\s]?login|access[_\s]?denied|authentication[_\s]?failed)\b',
                r'\b(unusual[_\s]?activity|anomaly|suspicious|warning|alert)\b',
                r'\b(after[_\s]?hours|weekend[_\s]?access|elevated[_\s]?privileges)\b'
            ],
            'file_indicators': [
                r'\.(exe|bat|cmd|scr|pif|com|vbs|js|jar|dll)$',
                r'\b(encoded|encrypted|obfuscated|packed|compressed)\b',
                r'\b(temporary|temp|cache|hidden|system)\b'
            ],
            'ip_patterns': [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                r'\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){7}\b'  # IPv6
            ],
            'hash_patterns': [
                r'\b[a-fA-F0-9]{32}\b',  # MD5
                r'\b[a-fA-F0-9]{40}\b',  # SHA1
                r'\b[a-fA-F0-9]{64}\b'   # SHA256
            ]
        }
        
        # Compile regex patterns
        compiled_patterns = {}
        for category, pattern_list in patterns.items():
            compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
            
        return compiled_patterns
    
    def load_bert_model(self):
        """Load BERT tokenizer and model"""
        print("🤖 Loading Enhanced BERT AI Model...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        logger.info(f"Loading BERT model: {self.model_name}")
        
        print("📥 Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print("🧠 Loading neural network weights...")
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        print("✅ Enhanced BERT model loaded successfully!")
        
    def extract_statistical_features(self, texts):
        """
        Extract statistical and linguistic features from text
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of statistical features
        """
        print("📊 Extracting statistical text features...")
        
        features = []
        for text in texts:
            feature_vector = []
            
            # Basic text statistics
            feature_vector.extend([
                len(text),                          # Text length
                len(text.split()),                  # Word count
                len(set(text.split())),            # Unique word count
                text.count('\n'),                   # Line count
                text.count('.'),                    # Sentence count (approx)
                text.count('!') + text.count('?'),  # Exclamation/question count
            ])
            
            # Character frequency analysis
            char_counts = Counter(text.lower())
            total_chars = len(text)
            if total_chars > 0:
                feature_vector.extend([
                    char_counts.get('e', 0) / total_chars,  # English frequency
                    char_counts.get('t', 0) / total_chars,
                    char_counts.get('a', 0) / total_chars,
                    char_counts.get('o', 0) / total_chars,
                    char_counts.get('i', 0) / total_chars,
                    char_counts.get('n', 0) / total_chars,
                ])
            else:
                feature_vector.extend([0] * 6)
            
            # Readability metrics
            try:
                feature_vector.extend([
                    textstat.flesch_reading_ease(text),
                    textstat.flesch_kincaid_grade(text),
                    textstat.automated_readability_index(text),
                ])
            except:
                feature_vector.extend([0, 0, 0])
            
            # Pattern matching counts
            pattern_counts = []
            for category, patterns in self.forensic_patterns.items():
                count = sum(len(pattern.findall(text)) for pattern in patterns)
                pattern_counts.append(count)
            feature_vector.extend(pattern_counts)
            
            # Entropy calculation (randomness measure)
            if len(text) > 0:
                entropy = -sum((count/len(text)) * np.log2(count/len(text)) 
                              for count in char_counts.values() if count > 0)
                feature_vector.append(entropy)
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
        
        print(f"✅ Generated {len(features)} statistical feature vectors with {len(features[0])} dimensions")
        return np.array(features)
    
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
        
        embeddings = []
        batch_size = 8  # Process in batches for efficiency
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            if (i // batch_size + 1) % 5 == 0 or i == 0:
                print(f"   Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            ).to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        print(f"✅ Generated {len(embeddings)} BERT embeddings of size {embeddings[0].shape[0]} dimensions!")
        return np.array(embeddings)
    
    def extract_tfidf_features(self, texts, max_features=5000):
        """
        Extract TF-IDF features from texts
        
        Args:
            texts: List of text strings
            max_features: Maximum number of TF-IDF features
            
        Returns:
            numpy array of TF-IDF features
        """
        print(f"📝 Extracting TF-IDF features (max_features={max_features})...")
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        print(f"✅ Generated TF-IDF features with shape: {tfidf_features.shape}")
        return tfidf_features
    
    def prepare_enhanced_training_data(self, data_path='data/processed'):
        """
        Prepare enhanced training data with multiple feature types
        
        Returns:
            Combined feature matrix and labels
        """
        print("🔍 Preparing enhanced forensic training data...")
        logger.info("Preparing enhanced training data...")
        
        # Load processed text files
        data_dir = Path(data_path)
        texts = []
        labels = []
        
        print(f"📂 Scanning directory: {data_path}")
        file_list = list(data_dir.glob('*.txt'))
        print(f"📄 Found {len(file_list)} forensic files to analyze")
        
        # Load and label data
        for i, file_path in enumerate(file_list):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"   📖 Reading file {i+1}/{len(file_list)}: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Enhanced heuristic labeling
            label = self._enhanced_classify_content_heuristic(content, file_path.name)
            
            texts.append(content[:3000])  # Increased text length for better context
            labels.append(label)
            
        print(f"✅ Loaded {len(texts)} forensic samples for training")
        
        # Show label distribution
        label_counts = {0: 0, 1: 0, 2: 0}
        for label in labels:
            label_counts[label] += 1
        
        print("📊 Enhanced Label Distribution:")
        print(f"   🟢 BENIGN: {label_counts[0]} files ({label_counts[0]/len(labels)*100:.1f}%)")
        print(f"   🟡 SUSPICIOUS: {label_counts[1]} files ({label_counts[1]/len(labels)*100:.1f}%)")
        print(f"   🔴 MALICIOUS: {label_counts[2]} files ({label_counts[2]/len(labels)*100:.1f}%)")
        
        # Extract multiple feature types
        print("\n🚀 Extracting multiple feature types...")
        
        # 1. BERT embeddings (768 dimensions)
        bert_features = self.extract_bert_embeddings(texts)
        
        # 2. TF-IDF features (5000 dimensions)
        tfidf_features = self.extract_tfidf_features(texts)
        
        # 3. Statistical features (~20 dimensions)
        stat_features = self.extract_statistical_features(texts)
        
        # Combine all features
        print("🔗 Combining feature types...")
        X_combined = np.hstack([bert_features, tfidf_features, stat_features])
        
        print(f"📊 Combined feature matrix shape: {X_combined.shape}")
        print(f"   - BERT features: {bert_features.shape[1]} dims")
        print(f"   - TF-IDF features: {tfidf_features.shape[1]} dims") 
        print(f"   - Statistical features: {stat_features.shape[1]} dims")
        print(f"   - Total: {X_combined.shape[1]} dimensions")
        
        y = np.array(labels)
        
        return X_combined, y
    
    def _enhanced_classify_content_heuristic(self, content, filename):
        """
        Enhanced heuristic-based labeling with pattern matching
        """
        content_lower = content.lower()
        
        # Pattern-based scoring
        pattern_scores = {}
        for category, patterns in self.forensic_patterns.items():
            count = sum(len(pattern.findall(content)) for pattern in patterns)
            pattern_scores[category] = count
        
        # Calculate weighted threat score
        malicious_score = (
            pattern_scores.get('malware_signatures', 0) * 3 +
            pattern_scores.get('attack_patterns', 0) * 3 +
            pattern_scores.get('network_threats', 0) * 2 +
            pattern_scores.get('hash_patterns', 0) * 1 +
            pattern_scores.get('ip_patterns', 0) * 0.5
        )
        
        suspicious_score = (
            pattern_scores.get('suspicious_activities', 0) * 2 +
            pattern_scores.get('file_indicators', 0) * 1 +
            pattern_scores.get('ip_patterns', 0) * 1
        )
        
        # Enhanced classification logic
        if malicious_score >= 5:
            print(f"   🚨 HIGH THREAT MALICIOUS detected in {filename[:20]}... (score: {malicious_score})")
            return 2  # MALICIOUS
        elif malicious_score >= 3:
            print(f"   🚨 MALICIOUS detected in {filename[:20]}... (score: {malicious_score})")
            return 2  # MALICIOUS
        elif malicious_score >= 2 and suspicious_score >= 3:
            print(f"   ⚠️  ELEVATED SUSPICIOUS detected in {filename[:20]}... (M:{malicious_score}, S:{suspicious_score})")
            return 1  # SUSPICIOUS
        elif suspicious_score >= 4:
            print(f"   ⚠️  HIGH SUSPICIOUS detected in {filename[:20]}... (score: {suspicious_score})")
            return 1  # SUSPICIOUS
        elif suspicious_score >= 2 or malicious_score >= 1:
            print(f"   ⚠️  SUSPICIOUS detected in {filename[:20]}... (M:{malicious_score}, S:{suspicious_score})")
            return 1  # SUSPICIOUS
        else:
            return 0  # BENIGN
    
    def train_enhanced_classifier(self, X, y, test_size=0.2):
        """
        Train enhanced ensemble classifier with hyperparameter tuning
        """
        print("🚀 Training Enhanced Multi-Model Ensemble...")
        print(f"   📊 Training data shape: {X.shape}")
        print(f"   📋 Labels: {len(y)} samples")
        logger.info("Training enhanced ensemble classifier...")
        
        # Scale features
        print("⚖️  Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification
        print("✂️  Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"   🎯 Train set: {len(X_train)} samples")
        print(f"   🧪 Test set: {len(X_test)} samples")
        
        # Create enhanced ensemble with multiple algorithms
        print("🌳 Building Multi-Model Ensemble...")
        
        # XGBoost with tuned parameters
        xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Random Forest with tuned parameters
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Create voting ensemble
        self.ensemble_classifier = VotingClassifier(
            estimators=[
                ('xgb', xgb_classifier),
                ('rf', rf_classifier)
            ],
            voting='soft'  # Use probability averaging
        )
        
        print("💪 Training ensemble models...")
        self.ensemble_classifier.fit(X_train, y_train)
        print("✅ Enhanced training completed!")
        
        # Cross-validation for robust evaluation
        print("🔍 Performing cross-validation...")
        cv_scores = cross_val_score(self.ensemble_classifier, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluate on test set
        print("🧪 Evaluating enhanced model performance...")
        y_pred = self.ensemble_classifier.predict(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n🎉 ENHANCED MODEL PERFORMANCE METRICS:")
        print("=" * 60)
        print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎪 Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"🎭 Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"🎨 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"📊 CV Score:  {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        
        # Check if target accuracy achieved
        if accuracy >= 0.85:
            print(f"🎉 TARGET ACCURACY ACHIEVED! ({accuracy*100:.2f}% >= 85%)")
        else:
            print(f"⚠️  Target accuracy not yet reached. Current: {accuracy*100:.2f}%, Target: 85%")
        
        # Store enhanced metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'feature_dimensions': X.shape[1]
        }
        
        logger.info("Enhanced Classification Report:")
        print("\n📊 DETAILED ENHANCED CLASSIFICATION REPORT:")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=list(self.labels.values())))
        
        # Generate enhanced visualizations
        self.create_enhanced_confusion_matrix(y_test, y_pred)
        self.create_enhanced_metrics_visualization()
        
        return X_test, y_test, y_pred
    
    def predict_enhanced(self, texts):
        """
        Enhanced prediction with multiple feature types
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predictions with enhanced confidence scores
        """
        if self.ensemble_classifier is None:
            raise ValueError("Enhanced classifier not trained. Call train_enhanced_classifier() first.")
            
        print(f"🔮 Making enhanced predictions for {len(texts)} texts...")
        
        # Extract all feature types
        bert_features = self.extract_bert_embeddings(texts)
        tfidf_features = self.extract_tfidf_features(texts)
        stat_features = self.extract_statistical_features(texts)
        
        # Combine features
        X_combined = np.hstack([bert_features, tfidf_features, stat_features])
        X_scaled = self.scaler.transform(X_combined)
        
        # Predict with ensemble
        predictions = self.ensemble_classifier.predict(X_scaled)
        probabilities = self.ensemble_classifier.predict_proba(X_scaled)
        
        results = []
        for i, pred in enumerate(predictions):
            confidence = float(np.max(probabilities[i]))
            results.append({
                'label': self.labels[pred],
                'confidence': confidence,
                'probabilities': {
                    self.labels[j]: float(probabilities[i][j]) 
                    for j in range(len(self.labels))
                },
                'enhanced_score': confidence * 100  # Enhanced scoring
            })
            
        return results
    
    def create_enhanced_confusion_matrix(self, y_test, y_pred):
        """
        Create enhanced confusion matrix visualization
        """
        print("📈 Creating enhanced confusion matrix visualization...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure with high DPI
        plt.figure(figsize=(12, 10), dpi=300)
        
        # Create enhanced heatmap
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='plasma',
                   xticklabels=list(self.labels.values()),
                   yticklabels=list(self.labels.values()),
                   cbar_kws={'label': 'Number of Samples'},
                   square=True,
                   linewidths=0.5)
        
        plt.title('ForensiQ Enhanced Evidence Classification\nConfusion Matrix (Multi-Feature Ensemble)', 
                 fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold')
        plt.ylabel('True Labels', fontsize=14, fontweight='bold')
        
        # Add enhanced accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.5, 0.02, f'Enhanced Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | Target: 85%+', 
                   ha='center', fontsize=14, fontweight='bold', 
                   color='green' if accuracy >= 0.85 else 'orange')
        
        # Ensure directory exists
        reports_dir = Path('reports/screenshots')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced plots
        matrix_path = reports_dir / 'enhanced_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Enhanced confusion matrix saved to: {matrix_path}")
        
        pdf_path = reports_dir / 'enhanced_confusion_matrix.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"   📄 Enhanced PDF version saved to: {pdf_path}")
        
        plt.close()
        
        return cm
    
    def create_enhanced_metrics_visualization(self):
        """
        Create comprehensive enhanced metrics visualization
        """
        if not hasattr(self, 'metrics'):
            print("⚠️  No metrics available. Train the model first.")
            return
            
        print("📊 Creating enhanced performance metrics visualization...")
        
        # Create comprehensive metrics dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        
        # 1. Metrics comparison bar chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            self.metrics['accuracy'],
            self.metrics['precision'], 
            self.metrics['recall'],
            self.metrics['f1_score']
        ]
        
        colors = ['#2E8B57' if v >= 0.85 else '#FF6B35' if v < 0.7 else '#FFD23F' for v in metrics_values]
        bars1 = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
        ax1.set_title('Enhanced Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (85%)')
        
        # Add value labels on bars
        for bar, value in zip(bars1, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-validation results
        if 'cv_accuracy_mean' in self.metrics:
            cv_mean = self.metrics['cv_accuracy_mean']
            cv_std = self.metrics['cv_accuracy_std']
            
            ax2.bar(['CV Accuracy'], [cv_mean], 
                   color='#4CAF50' if cv_mean >= 0.85 else '#FFA726',
                   alpha=0.8)
            ax2.errorbar(['CV Accuracy'], [cv_mean], yerr=[cv_std], 
                        color='black', capsize=5, capthick=2)
            ax2.set_title('Cross-Validation Results', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Accuracy Score', fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (85%)')
            ax2.text(0, cv_mean + cv_std + 0.02, f'{cv_mean:.3f} ± {cv_std:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance (if available)
        feature_info = [
            ('BERT Embeddings', 768, '#FF6B35'),
            ('TF-IDF Features', 5000, '#4ECDC4'), 
            ('Statistical Features', self.metrics.get('feature_dimensions', 0) - 768 - 5000, '#45B7D1')
        ]
        
        feature_names = [info[0] for info in feature_info]
        feature_counts = [info[1] for info in feature_info]
        feature_colors = [info[2] for info in feature_info]
        
        ax3.pie(feature_counts, labels=feature_names, colors=feature_colors, autopct='%1.1f%%')
        ax3.set_title('Feature Distribution in Enhanced Model', fontsize=14, fontweight='bold')
        
        # 4. Accuracy progression
        baseline_accuracy = 0.6667  # Original model accuracy
        enhanced_accuracy = self.metrics['accuracy']
        
        progression = ['Baseline Model', 'Enhanced Model']
        accuracies = [baseline_accuracy, enhanced_accuracy]
        colors_prog = ['#FF6B35', '#2E8B57' if enhanced_accuracy >= 0.85 else '#FFD23F']
        
        bars4 = ax4.bar(progression, accuracies, color=colors_prog, alpha=0.8)
        ax4.set_title('Accuracy Improvement', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Accuracy Score', fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (85%)')
        
        # Add improvement arrow
        improvement = enhanced_accuracy - baseline_accuracy
        ax4.annotate(f'+{improvement:.3f}\n({improvement*100:.1f}%)', 
                    xy=(1, enhanced_accuracy), xytext=(0.5, enhanced_accuracy + 0.1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    ha='center', fontweight='bold', fontsize=12)
        
        # Add value labels
        for bar, value in zip(bars4, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save enhanced metrics visualization
        reports_dir = Path('reports/screenshots')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = reports_dir / 'enhanced_metrics_dashboard.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Enhanced metrics dashboard saved to: {metrics_path}")
        
        pdf_path = reports_dir / 'enhanced_metrics_dashboard.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"   📄 Enhanced PDF dashboard saved to: {pdf_path}")
        
        plt.close()
    
    def save_enhanced_model(self, model_path='models/classifier/enhanced_evidence_classifier.pkl'):
        """
        Save the enhanced trained model and components
        """
        print("💾 Saving enhanced model components...")
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble classifier
        joblib.dump(self.ensemble_classifier, model_path)
        
        # Save TF-IDF vectorizer
        tfidf_path = model_dir / 'enhanced_tfidf_vectorizer.pkl'
        joblib.dump(self.tfidf_vectorizer, tfidf_path)
        
        # Save scaler
        scaler_path = model_dir / 'enhanced_feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save metrics
        metrics_path = model_dir / 'enhanced_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Enhanced model saved to: {model_path}")
        print(f"✅ TF-IDF vectorizer saved to: {tfidf_path}")
        print(f"✅ Feature scaler saved to: {scaler_path}")
        print(f"✅ Metrics saved to: {metrics_path}")


def main():
    """
    Main function to run enhanced evidence classification
    """
    print("🚀 ForensiQ Enhanced Evidence Classification Engine")
    print("=" * 60)
    print("🎯 Target Accuracy: 85%+")
    print("🤖 Multi-Feature Ensemble: BERT + TF-IDF + Statistical")
    print("=" * 60)
    
    # Initialize enhanced classifier
    classifier = EnhancedEvidenceClassifier()
    
    # Prepare enhanced training data
    print("\n📊 Phase 1: Enhanced Data Preparation")
    print("-" * 40)
    X, y = classifier.prepare_enhanced_training_data()
    
    # Train enhanced classifier
    print("\n🚀 Phase 2: Enhanced Model Training")
    print("-" * 40)
    X_test, y_test, y_pred = classifier.train_enhanced_classifier(X, y)
    
    # Save enhanced model
    print("\n💾 Phase 3: Model Persistence")
    print("-" * 40)
    classifier.save_enhanced_model()
    
    # Save enhanced evaluation report
    print("\n📄 Phase 4: Enhanced Evaluation Report")
    print("-" * 40)
    
    enhanced_report = {
        "model_info": {
            "model_name": "Enhanced Multi-Feature Ensemble",
            "base_model": "bert-base-uncased",
            "algorithms": ["XGBoost", "RandomForest"],
            "feature_types": ["BERT", "TF-IDF", "Statistical"],
            "training_samples": len(X),
            "test_samples": len(X_test),
            "total_feature_dimensions": X.shape[1]
        },
        "enhanced_metrics": classifier.metrics,
        "target_achievement": {
            "target_accuracy": 0.85,
            "achieved_accuracy": classifier.metrics['accuracy'],
            "target_met": classifier.metrics['accuracy'] >= 0.85,
            "improvement_from_baseline": classifier.metrics['accuracy'] - 0.6667
        }
    }
    
    report_path = Path('reports/enhanced_model_evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(enhanced_report, f, indent=2)
    
    print(f"✅ Enhanced evaluation report saved to: {report_path}")
    
    # Demo enhanced predictions
    print("\n🔮 Phase 5: Enhanced Prediction Demo")
    print("-" * 40)
    
    demo_texts = [
        "System malware detected. Virus signature found in executable file. Immediate quarantine required.",
        "User login failed multiple times from suspicious IP address. Potential brute force attack.",
        "Regular system backup completed successfully. All files verified."
    ]
    
    predictions = classifier.predict_enhanced(demo_texts)
    
    print("🎭 Enhanced Prediction Results:")
    for i, (text, pred) in enumerate(zip(demo_texts, predictions)):
        print(f"\n   Text {i+1}: {text[:50]}...")
        print(f"   🏷️  Label: {pred['label']}")
        print(f"   🎯 Enhanced Score: {pred['enhanced_score']:.1f}%")
        print(f"   📊 Confidence: {pred['confidence']:.3f}")
    
    # Final summary
    print(f"\n🎉 ENHANCED CLASSIFICATION ENGINE COMPLETE!")
    print("=" * 60)
    final_accuracy = classifier.metrics['accuracy']
    print(f"🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    if final_accuracy >= 0.85:
        print("✅ TARGET ACCURACY ACHIEVED! (85%+)")
    else:
        print(f"⚠️  Target not reached. Improvement: {(final_accuracy-0.6667)*100:.1f}% points")
    print("=" * 60)


if __name__ == "__main__":
    main()
