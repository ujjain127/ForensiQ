"""
ForensiQ Evidence Classification Engine
======================================
Classifies digital evidence as Benign, Suspicious, or Malicious
Uses BERT embeddings + XGBoost classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import torch
import joblib
import logging
from pathlib import Path

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
        logger.info(f"Loading BERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
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
            
        embeddings = []
        
        for text in texts:
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
                
        return np.array(embeddings)
    
    def prepare_training_data(self, data_path='data/processed'):
        """
        Prepare training data from processed forensic files
        
        Returns:
            X: BERT embeddings
            y: Classification labels
        """
        logger.info("Preparing training data...")
        
        # Load processed text files
        data_dir = Path(data_path)
        texts = []
        labels = []
        
        # Load and label data based on file patterns or content analysis
        for file_path in data_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Simple heuristic labeling (enhance with domain expertise)
            label = self._classify_content_heuristic(content, file_path.name)
            
            texts.append(content[:2000])  # Limit text length
            labels.append(label)
            
        logger.info(f"Loaded {len(texts)} samples for training")
        
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
        
        # Malicious indicators
        malicious_keywords = [
            'malware', 'virus', 'trojan', 'backdoor', 'exploit',
            'attack', 'breach', 'unauthorized', 'intrusion',
            'password crack', 'keylogger', 'ransomware'
        ]
        
        # Suspicious indicators  
        suspicious_keywords = [
            'suspicious', 'unusual', 'anomaly', 'warning',
            'failed login', 'access denied', 'error',
            'connection refused', 'timeout'
        ]
        
        malicious_count = sum(1 for keyword in malicious_keywords if keyword in content_lower)
        suspicious_count = sum(1 for keyword in suspicious_keywords if keyword in content_lower)
        
        if malicious_count >= 2:
            return 2  # MALICIOUS
        elif suspicious_count >= 2 or malicious_count >= 1:
            return 1  # SUSPICIOUS
        else:
            return 0  # BENIGN
            
    def train_classifier(self, X, y, test_size=0.2):
        """
        Train XGBoost classifier on BERT embeddings
        """
        logger.info("Training XGBoost classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train XGBoost
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.xgb_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.xgb_classifier.predict(X_test)
        
        logger.info("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=list(self.labels.values())))
        
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
    
    def save_model(self, model_dir='models/classifier'):
        """Save trained models"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost classifier
        joblib.dump(self.xgb_classifier, model_path / 'xgb_classifier.joblib')
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'labels': self.labels
        }
        joblib.dump(metadata, model_path / 'metadata.joblib')
        
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
    # Initialize classifier
    classifier = EvidenceClassifier()
    
    # Prepare training data
    X, y = classifier.prepare_training_data()
    
    # Train classifier
    X_test, y_test, y_pred = classifier.train_classifier(X, y)
    
    # Save model
    classifier.save_model()
    
    # Example prediction
    sample_texts = [
        "Normal system startup completed successfully",
        "Multiple failed login attempts detected from unknown IP",
        "Malware signature detected in downloaded file"
    ]
    
    predictions = classifier.predict(sample_texts)
    
    print("\n🔍 Sample Predictions:")
    for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
        print(f"{i+1}. Text: {text[:50]}...")
        print(f"   Label: {pred['label']} (Confidence: {pred['confidence']:.3f})")
        print()

if __name__ == "__main__":
    main()
