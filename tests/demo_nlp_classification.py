"""
ForensiQ NLP Classification Demo
===============================
Interactive demo showing how the NLP analyzer classifies different types of forensic content
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nlp_analysis.analyze_text import ForensicNLPAnalyzer

def demo_classification():
    """Demonstrate NLP classification with real-world examples"""
    
    analyzer = ForensicNLPAnalyzer()
    
    # Real-world examples
    examples = [
        {
            'title': '💰 PHISHING EMAIL SCAM',
            'content': '''
            Your bank account security@fakebank.tk will be suspended unless you verify immediately.
            Login at http://secure-bank.tk/login with your password and send $500 bitcoin to wallet:
            1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
            Contact +1-800-FAKE-123 urgent!
            '''
        },
        {
            'title': '🔒 CYBERSECURITY BREACH',
            'content': '''
            Alert: Unauthorized access detected on server 10.0.1.50
            Admin user "root" logged in from suspicious IP 45.67.89.123
            Malware hash: d41d8cd98f00b204e9800998ecf8427e
            Password database compromised - immediate action required
            '''
        },
        {
            'title': '💬 SUSPICIOUS CHAT LOG',
            'content': '''
            [2023-07-23 14:30] dealer123: Got the package ready
            [2023-07-23 14:31] buyer456: How much for 5 units?
            [2023-07-23 14:32] dealer123: $100 each, cash only
            [2023-07-23 14:33] buyer456: Meet at usual spot tonight
            '''
        },
        {
            'title': '⚡ THREAT MESSAGE',
            'content': '''
            I know where you live at 123 Main Street. 
            You better watch out or I will hurt you and your family.
            This is not a joke - I have weapons and I will use them.
            Pay me $1000 or face the consequences.
            '''
        },
        {
            'title': '📋 NORMAL BUSINESS EMAIL',
            'content': '''
            Hi team, please review the quarterly reports by Friday.
            The meeting is scheduled for 2 PM in conference room A.
            Let me know if you have any questions.
            Best regards, Manager
            '''
        }
    ]
    
    print("🔍 ForensiQ NLP Classification Demo")
    print("=" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\\n{example['title']}")
        print("-" * 50)
        
        # Analyze content
        entities = analyzer.extract_entities(example['content'])
        sentiment = analyzer.analyze_sentiment(example['content'])
        keywords = analyzer.extract_forensic_keywords(example['content'])
        suspicious_urls = analyzer.analyze_suspicious_urls(example['content'])
        
        # Risk assessment
        risk_level = sentiment['risk_level']
        risk_emoji = "🚨" if risk_level == "HIGH" else "⚠️" if risk_level == "MEDIUM" else "✅"
        
        print(f"{risk_emoji} RISK LEVEL: {risk_level}")
        print(f"📊 SENTIMENT SCORE: {sentiment['sentiment_score']:.3f}")
        
        # Forensic categories
        if keywords:
            print(f"🔍 FORENSIC CATEGORIES:")
            for category, items in keywords.items():
                print(f"   • {category.replace('_', ' ').title()}: {len(items)} matches")
        
        # Entities found
        entity_count = sum(len(items) for items in entities.values())
        if entity_count > 0:
            print(f"📋 ENTITIES EXTRACTED: {entity_count} total")
            for entity_type, items in entities.items():
                if items:
                    print(f"   • {entity_type}: {[item['text'] for item in items[:3]]}")
        
        # Suspicious URLs
        if suspicious_urls:
            print(f"⚠️  SUSPICIOUS URLS: {len(suspicious_urls)} found")
            for url_info in suspicious_urls:
                print(f"   • {url_info['url']} (Risk: {url_info['risk_level']})")
        
        # Overall assessment
        if risk_level == "HIGH":
            print("🚨 **REQUIRES IMMEDIATE INVESTIGATION**")
        elif risk_level == "MEDIUM":
            print("⚠️  **FLAGGED FOR REVIEW**")
        else:
            print("✅ **APPEARS NORMAL**")
    
    print("\\n" + "=" * 60)
    print("📝 SUMMARY:")
    print("✅ ForensiQ NLP successfully identifies:")
    print("   • Financial fraud indicators")
    print("   • Cybersecurity threats")
    print("   • Drug trafficking communications")
    print("   • Threatening language")
    print("   • Suspicious URLs and entities")
    print("   • Risk levels for prioritization")

if __name__ == "__main__":
    demo_classification()
