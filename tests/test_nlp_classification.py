"""
ForensiQ NLP Classification Test Suite
=====================================
Test the NLP analyzer with various forensic scenarios
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nlp_analysis.analyze_text import ForensicNLPAnalyzer
import json

def test_forensic_scenarios():
    """Test the NLP analyzer with realistic forensic scenarios"""
    
    # Initialize analyzer (will work without spaCy for basic features)
    analyzer = ForensicNLPAnalyzer()
    
    # Test scenarios with expected classifications
    test_cases = [
        {
            'name': 'Financial Fraud Email',
            'text': '''
            Subject: Urgent Payment Required
            
            Dear Sir,
            
            We need you to transfer $50,000 to bitcoin wallet bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh immediately.
            This is for the investment opportunity we discussed. Wire transfer must be completed today.
            Please send your bank account details and credit card information to process.
            Your account has been compromised, please login with admin password to verify.
            
            Contact: scammer@fraudulent-bank.tk
            Phone: +1-555-SCAM-123
            ''',
            'expected_categories': ['financial_fraud', 'cybersecurity'],
            'expected_risk': 'HIGH'
        },
        {
            'name': 'Cybersecurity Incident',
            'text': '''
            System breach detected at 192.168.1.50. Unauthorized login attempts using admin@company.com.
            Malware signature found: trojan.ransomware.xyz
            Backup systems compromised. Password hash dump detected.
            Attacker IP: 203.45.67.89
            Vulnerability exploited: CVE-2023-1234
            ''',
            'expected_categories': ['cybersecurity'],
            'expected_risk': 'HIGH'
        },
        {
            'name': 'Threatening Messages',
            'text': '''
            I'm going to kill you and destroy everything you care about.
            You better watch your back. I know where you live at 123 Main St.
            This is not a joke - I will hurt you and your family.
            Meet me tonight or face the consequences.
            ''',
            'expected_categories': ['threat_language'],
            'expected_risk': 'HIGH'
        },
        {
            'name': 'Drug Trafficking Chat',
            'text': '''
            2023-01-15 10:30 dealer_mike: Got the new shipment of product
            2023-01-15 10:31 buyer_jay: How much for 2 packages?
            2023-01-15 10:32 dealer_mike: $500 each. High quality cocaine from supplier
            2023-01-15 10:33 buyer_jay: Meet at usual spot tonight. Bring the drugs
            2023-01-15 10:35 dealer_mike: Cash only. No questions asked
            ''',
            'expected_categories': ['drugs_trafficking'],
            'expected_risk': 'HIGH'
        },
        {
            'name': 'Normal Business Email',
            'text': '''
            Subject: Quarterly Meeting Schedule
            
            Hi team,
            
            Our quarterly meeting is scheduled for next Tuesday at 2 PM.
            Please review the attached reports and come prepared with your updates.
            
            Best regards,
            John Manager
            john.manager@company.com
            Phone: +1-555-WORK-123
            ''',
            'expected_categories': [],
            'expected_risk': 'LOW'
        }
    ]
    
    print("🧪 ForensiQ NLP Classification Test Suite")
    print("=" * 60)
    
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{total_tests}: {test_case['name']}")
        print("-" * 40)
        
        # Analyze the text
        entities = analyzer.extract_entities(test_case['text'])
        sentiment = analyzer.analyze_sentiment(test_case['text'])
        keywords = analyzer.extract_forensic_keywords(test_case['text'])
        suspicious_urls = analyzer.analyze_suspicious_urls(test_case['text'])
        
        # Check results
        found_categories = list(keywords.keys())
        risk_level = sentiment['risk_level']
        
        print(f"📊 Analysis Results:")
        print(f"   Risk Level: {risk_level}")
        print(f"   Forensic Categories: {found_categories}")
        print(f"   Entities Found: {sum(len(items) for items in entities.values())}")
        print(f"   Suspicious URLs: {len(suspicious_urls)}")
        
        # Validation
        test_passed = True
        
        # Check risk level
        if risk_level != test_case['expected_risk']:
            print(f"❌ Risk level mismatch: expected {test_case['expected_risk']}, got {risk_level}")
            test_passed = False
        else:
            print(f"✅ Risk level correct: {risk_level}")
        
        # Check categories
        expected_cats = set(test_case['expected_categories'])
        found_cats = set(found_categories)
        
        if not expected_cats.issubset(found_cats) and expected_cats:
            missing = expected_cats - found_cats
            print(f"❌ Missing categories: {missing}")
            test_passed = False
        elif expected_cats or found_cats:
            print(f"✅ Categories detected correctly: {found_cats}")
        else:
            print(f"✅ No suspicious categories (as expected)")
        
        # Show detailed findings
        if entities.get('EMAIL'):
            print(f"📧 Emails: {[e['text'] for e in entities['EMAIL']]}")
        if entities.get('PHONE'):
            print(f"📞 Phones: {[p['text'] for p in entities['PHONE']]}")
        if entities.get('IP_ADDRESS'):
            print(f"🌐 IP Addresses: {[ip['text'] for ip in entities['IP_ADDRESS']]}")
        if suspicious_urls:
            print(f"⚠️  Suspicious URLs: {[url['url'] for url in suspicious_urls]}")
        
        if test_passed:
            passed_tests += 1
            print("✅ Test PASSED")
        else:
            print("❌ Test FAILED")
    
    print("\n" + "=" * 60)
    print(f"📈 Test Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! NLP classification is working correctly.")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Review the implementation.")
    
    return passed_tests == total_tests

def test_entity_extraction():
    """Test entity extraction specifically"""
    
    analyzer = ForensicNLPAnalyzer()
    
    test_text = """
    Contact John Smith at john.smith@example.com or +1-555-123-4567.
    His IP address is 192.168.1.100 and he uses credit card 4532-1234-5678-9012.
    Visit https://suspicious-site.tk/malware or bit.ly/hack123 for more info.
    """
    
    print("\n🔍 Entity Extraction Test")
    print("-" * 30)
    
    entities = analyzer.extract_entities(test_text)
    
    expected_entities = {
        'EMAIL': 1,
        'PHONE': 1, 
        'IP_ADDRESS': 1,
        'CREDIT_CARD': 1,
        'URL': 1
    }
    
    print("Found entities:")
    for entity_type, items in entities.items():
        if items:
            print(f"  {entity_type}: {[item['text'] for item in items]} ({len(items)} found)")
    
    # Validate
    all_correct = True
    for entity_type, expected_count in expected_entities.items():
        actual_count = len(entities.get(entity_type, []))
        if actual_count < expected_count:
            print(f"❌ {entity_type}: expected at least {expected_count}, got {actual_count}")
            all_correct = False
        else:
            print(f"✅ {entity_type}: {actual_count} found (expected {expected_count})")
    
    return all_correct

if __name__ == "__main__":
    print("🚀 Starting ForensiQ NLP Tests...")
    
    # Test main classification
    classification_passed = test_forensic_scenarios()
    
    # Test entity extraction
    entity_passed = test_entity_extraction()
    
    print("\n" + "=" * 60)
    if classification_passed and entity_passed:
        print("🎉 ALL TESTS PASSED! ForensiQ NLP is ready for use.")
    else:
        print("❌ Some tests failed. Check the implementation.")
        
    print("\n💡 Note: Full spaCy NER requires 'python -m spacy download en_core_web_sm'")
    print("   Current tests use regex-based entity extraction.")
