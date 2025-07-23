"""
ForensiQ NLP Analysis Module
============================
Named Entity Recognition, sentiment analysis, and keyword extraction
for chat logs, emails, and documents
"""

import spacy
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
import email
from email.parser import Parser
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForensicNLPAnalyzer:
    """
    Advanced NLP analyzer for digital forensics evidence
    """
    
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Forensic keywords for different categories
        self.forensic_keywords = {
            'financial_fraud': [
                'credit card', 'bank account', 'wire transfer', 'bitcoin', 'cryptocurrency',
                'money laundering', 'fraud', 'scam', 'investment', 'payment', 'transfer',
                'money', 'cash', 'wallet', '$', 'dollar'
            ],
            'cybersecurity': [
                'password', 'login', 'hack', 'breach', 'malware', 'virus', 'phishing',
                'ransomware', 'backdoor', 'exploit', 'vulnerability', 'attack', 'system',
                'admin', 'administrator', 'unauthorized', 'suspicious'
            ],
            'drugs_trafficking': [
                'drugs', 'cocaine', 'heroin', 'marijuana', 'trafficking', 'dealer',
                'supply', 'shipment', 'package', 'delivery', 'product'
            ],
            'terrorism': [
                'bomb', 'explosive', 'attack', 'target', 'operation', 'cell',
                'recruitment', 'training', 'weapon', 'plan'
            ],
            'child_exploitation': [
                'child', 'minor', 'underage', 'exploitation', 'abuse',
                'inappropriate', 'illegal content'
            ],
            'threat_language': [
                'kill', 'murder', 'death', 'threat', 'violence', 'harm',
                'destroy', 'revenge', 'punishment', 'hurt', 'die'
            ]
        }
        
        # Suspicious URL patterns
        self.suspicious_url_patterns = [
            r'bit\.ly/\w+',
            r'tinyurl\.com/\w+',
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
            r'[a-z]+\.tk/',
            r'[a-z]+\.ml/',
            r'[a-z]+\.ga/'
        ]
        
    def extract_entities(self, text):
        """
        Extract named entities from text using spaCy NER
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with extracted entities by category
        """
        if self.nlp is None:
            return {}
            
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities (countries, cities, states)
            'DATE': [],
            'MONEY': [],
            'EMAIL': [],
            'PHONE': [],
            'URL': [],
            'IP_ADDRESS': [],
            'CREDIT_CARD': []
        }
        
        # Extract spaCy entities
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores by default
                })
        
        # Extract custom entities using regex
        custom_entities = self._extract_custom_entities(text)
        for category, items in custom_entities.items():
            entities[category].extend(items)
            
        return entities
    
    def _extract_custom_entities(self, text):
        """Extract custom entities using regex patterns"""
        entities = {
            'EMAIL': [],
            'PHONE': [],
            'URL': [],
            'IP_ADDRESS': [],
            'CREDIT_CARD': []
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.finditer(email_pattern, text)
        for match in emails:
            entities['EMAIL'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Phone pattern
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.finditer(phone_pattern, text)
        for match in phones:
            entities['PHONE'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.90
            })
        
        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.finditer(url_pattern, text)
        for match in urls:
            entities['URL'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.98
            })
        
        # IP Address pattern
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.finditer(ip_pattern, text)
        for match in ips:
            # Validate IP address
            ip_parts = match.group().split('.')
            if all(0 <= int(part) <= 255 for part in ip_parts):
                entities['IP_ADDRESS'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })
        
        # Credit card pattern (basic)
        cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        cards = re.finditer(cc_pattern, text)
        for match in cards:
            entities['CREDIT_CARD'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.80
            })
        
        return entities
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment and emotional indicators
        (Basic implementation - enhance with specialized models)
        """
        text_lower = text.lower()
        
        # Emotional indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'quickly', 'hurry', 'rush', 'now', 'today']
        fear_words = ['scared', 'afraid', 'terrified', 'worried', 'nervous', 'anxious']
        anger_words = ['angry', 'furious', 'rage', 'hate', 'mad', 'pissed']
        threat_words = ['threat', 'kill', 'murder', 'destroy', 'hurt', 'harm', 'die', 'death', 'violence']
        
        scores = {
            'urgency': sum(1 for word in urgency_words if word in text_lower),
            'fear': sum(1 for word in fear_words if word in text_lower),
            'anger': sum(1 for word in anger_words if word in text_lower),
            'threat': sum(1 for word in threat_words if word in text_lower)
        }
        
        # Overall sentiment score (enhanced for forensic context)
        total_words = len(text_lower.split())
        if total_words > 0:
            # Weight threats heavily, consider urgency + financial terms as suspicious
            sentiment_score = (scores['threat'] * 5 + scores['anger'] * 3 + 
                             scores['fear'] * 2 + scores['urgency'] * 1) / total_words
            
            # Additional checks for high-risk content
            financial_terms = ['bitcoin', 'money', 'transfer', 'payment', 'bank', 'credit card']
            cyber_terms = ['hack', 'password', 'breach', 'malware', 'attack', 'admin']
            drug_terms = ['drugs', 'cocaine', 'dealer', 'trafficking', 'package', 'product']
            
            financial_score = sum(1 for term in financial_terms if term in text_lower)
            cyber_score = sum(1 for term in cyber_terms if term in text_lower)
            drug_score = sum(1 for term in drug_terms if term in text_lower)
            
            # Boost score if multiple risk categories present
            if financial_score > 0 and scores['urgency'] > 0:
                sentiment_score += 0.15  # Financial fraud indicator
            if cyber_score > 0:
                sentiment_score += 0.1   # Cybersecurity indicator
            if drug_score > 0:
                sentiment_score += 0.12  # Drug trafficking indicator
                
        else:
            sentiment_score = 0
            
        return {
            'emotional_indicators': scores,
            'sentiment_score': sentiment_score,
            'risk_level': 'HIGH' if sentiment_score > 0.08 else 'MEDIUM' if sentiment_score > 0.03 else 'LOW'
        }
    
    def extract_forensic_keywords(self, text):
        """
        Extract forensic keywords by category
        """
        text_lower = text.lower()
        found_keywords = {}
        
        for category, keywords in self.forensic_keywords.items():
            found = []
            for keyword in keywords:
                if keyword in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(keyword, start)
                        if pos == -1:
                            break
                        found.append({
                            'keyword': keyword,
                            'position': pos,
                            'context': text[max(0, pos-50):pos+len(keyword)+50]
                        })
                        start = pos + 1
            
            if found:
                found_keywords[category] = found
                
        return found_keywords
    
    def analyze_suspicious_urls(self, text):
        """
        Analyze URLs for suspicious patterns
        """
        suspicious_urls = []
        
        for pattern in self.suspicious_url_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                suspicious_urls.append({
                    'url': match.group(),
                    'pattern': pattern,
                    'position': match.start(),
                    'risk_level': 'HIGH' if '\\d+\\.' in pattern else 'MEDIUM'
                })
                
        return suspicious_urls
    
    def parse_email_content(self, email_path):
        """
        Parse email file and extract forensic information
        """
        try:
            with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            
            # Parse email
            msg = Parser().parsestr(email_content)
            
            # Extract email metadata
            email_data = {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'cc': msg.get('Cc', ''),
                'bcc': msg.get('Bcc', ''),
                'message_id': msg.get('Message-ID', ''),
                'body': ''
            }
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            email_data['body'] += body.decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True)
                if body:
                    email_data['body'] = body.decode('utf-8', errors='ignore')
            
            # Perform NLP analysis
            full_text = f"{email_data['subject']} {email_data['body']}"
            
            analysis = {
                'metadata': email_data,
                'entities': self.extract_entities(full_text),
                'sentiment': self.analyze_sentiment(email_data['body']),
                'forensic_keywords': self.extract_forensic_keywords(full_text),
                'suspicious_urls': self.analyze_suspicious_urls(full_text)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing email {email_path}: {e}")
            return None
    
    def analyze_chat_log(self, chat_text):
        """
        Analyze chat log for forensic indicators
        """
        # Split into individual messages (basic implementation)
        messages = chat_text.split('\n')
        
        analysis_results = []
        
        for i, message in enumerate(messages):
            if message.strip():
                # Extract timestamp if present
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})', message)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # Extract username/sender
                sender_match = re.search(r'^([^:]+):', message)
                sender = sender_match.group(1) if sender_match else 'unknown'
                
                # Extract message content
                content = message.split(':', 1)[-1].strip() if ':' in message else message
                
                # Analyze message
                msg_analysis = {
                    'message_id': i,
                    'timestamp': timestamp,
                    'sender': sender,
                    'content': content,
                    'entities': self.extract_entities(content),
                    'sentiment': self.analyze_sentiment(content),
                    'forensic_keywords': self.extract_forensic_keywords(content),
                    'suspicious_urls': self.analyze_suspicious_urls(content)
                }
                
                analysis_results.append(msg_analysis)
        
        return analysis_results
    
    def generate_nlp_summary(self, analysis_results):
        """
        Generate summary of NLP analysis findings
        """
        summary = {
            'total_entities': defaultdict(int),
            'risk_indicators': [],
            'forensic_categories': defaultdict(int),
            'sentiment_distribution': defaultdict(int),
            'suspicious_urls_count': 0
        }
        
        # Aggregate results
        for result in analysis_results:
            # Count entities
            if 'entities' in result:
                for entity_type, entities in result['entities'].items():
                    summary['total_entities'][entity_type] += len(entities)
            
            # Risk indicators
            if 'sentiment' in result and result['sentiment']['risk_level'] == 'HIGH':
                summary['risk_indicators'].append(result.get('message_id', 'unknown'))
            
            # Forensic keywords
            if 'forensic_keywords' in result:
                for category in result['forensic_keywords'].keys():
                    summary['forensic_categories'][category] += 1
            
            # Sentiment
            if 'sentiment' in result:
                summary['sentiment_distribution'][result['sentiment']['risk_level']] += 1
            
            # Suspicious URLs
            if 'suspicious_urls' in result:
                summary['suspicious_urls_count'] += len(result['suspicious_urls'])
        
        return dict(summary)

def main():
    """
    Example usage of ForensicNLPAnalyzer
    """
    analyzer = ForensicNLPAnalyzer()
    
    # Example text analysis
    sample_text = """
    John Smith sent an email to admin@company.com from his IP address 192.168.1.100.
    He mentioned transferring $50,000 to bitcoin wallet and was very angry about the delay.
    Contact him at +1-555-0123 or visit http://suspicious-site.tk/download.
    """
    
    print("🔍 ForensiQ NLP Analysis Demo")
    print("=" * 50)
    
    # Extract entities
    entities = analyzer.extract_entities(sample_text)
    print("\\n📋 Extracted Entities:")
    for entity_type, items in entities.items():
        if items:
            print(f"  {entity_type}: {[item['text'] for item in items]}")
    
    # Sentiment analysis
    sentiment = analyzer.analyze_sentiment(sample_text)
    print(f"\\n💭 Sentiment Analysis:")
    print(f"  Risk Level: {sentiment['risk_level']}")
    print(f"  Emotional Indicators: {sentiment['emotional_indicators']}")
    
    # Forensic keywords
    keywords = analyzer.extract_forensic_keywords(sample_text)
    print(f"\\n🚨 Forensic Keywords:")
    for category, found in keywords.items():
        print(f"  {category}: {[item['keyword'] for item in found]}")
    
    # Suspicious URLs
    urls = analyzer.analyze_suspicious_urls(sample_text)
    print(f"\\n🌐 Suspicious URLs:")
    for url_info in urls:
        print(f"  {url_info['url']} (Risk: {url_info['risk_level']})")

if __name__ == "__main__":
    main()
