import re

def clean_pii(raw_text):
    """
    Scrubs Personally Identifiable Information (PII) from resume text 
    to ensure unbiased AI evaluation.
    """
    print("Scrubbing PII for ethical AI evaluation...")
    
    # 1. Remove Email Addresses
    text = re.sub(r'\S+@\S+', '[REDACTED EMAIL]', raw_text)
    
    # 2. Remove Phone Numbers
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[REDACTED PHONE]', text)
    
    # 3. Remove URLs (LinkedIn, GitHub, Portfolios)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[REDACTED URL]', text)
    text = re.sub(r'www\.\S+', '[REDACTED URL]', text)
    text = re.sub(r'linkedin\.com/in/\S+', '[REDACTED LINKEDIN]', text)
    text = re.sub(r'github\.com/\S+', '[REDACTED GITHUB]', text)

    return text