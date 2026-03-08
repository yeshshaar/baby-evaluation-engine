import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load the hidden API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def parse_resume_with_llama(raw_text):
    """Uses Llama 3 to extract structured data from raw resume text."""
    print("Sending text to Llama 3 for analysis...")
    
    # This is where your Prompt Engineering skills shine
    system_prompt = """
    You are an expert AI Recruiter Assistant. Your job is to extract specific information from resumes.
    You must return ONLY a raw JSON object. Do not include markdown formatting, backticks, or conversational text.
    
    Extract the following fields:
    - name: string
    - years_of_experience: integer
    - core_skills: list of strings
    - tools: list of strings
    - soft_skills: list of strings
    - projects: list of strings
    
    If you cannot find a value, use null or an empty list [].
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", # The updated, supported model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the data from this resume:\n\n{raw_text}"}
            ],
            temperature=0.0, # We want precise facts, not creativity
        )
        
        # Extract the JSON string from the AI's response
        result = completion.choices[0].message.content
        return json.loads(result)
        
    except Exception as e:
        print(f"AI Extraction Failed: {e}")
        return None

# --- Testing the AI ---
if __name__ == "__main__":
    # We will import the function you wrote in the last step!
    from extractor import extract_text_from_pdf
    
    test_resume_path = "data/raw/test_resume.pdf"
    
    if os.path.exists(test_resume_path):
        raw_text = extract_text_from_pdf(test_resume_path)
        
        if raw_text:
            structured_data = parse_resume_with_llama(raw_text)
            print("\n--- AI EXTRACTION RESULTS ---")
            print(json.dumps(structured_data, indent=4))
    else:
        print("Please put test_resume.pdf in data/raw/")