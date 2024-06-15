import gradio as gr
import google.generativeai as genai
import os
import docx2txt
import PyPDF2 as pdf
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set up the API key
genai.configure(api_key="AIzaSyDuV0_YnMu0wYFLATzmmk0SdshQUQPDrhk")
os.environ["GOOGLE_API_KEY"]="AIzaSyDuV0_YnMu0wYFLATzmmk0SdshQUQPDrhk"

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Create a GenerativeModel instance
llm = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Prompt Template
input_prompt_template = """
As an experienced Applicant Tracking System (ATS) analyst,
with profound knowledge in technology, software engineering, data science, 
and big data engineering, your role involves evaluating resumes against job descriptions.
Recognizing the competitive job market, provide top-notch assistance for resume improvement.
Your goal is to analyze the resume against the given job description, 
assign a percentage match based on key criteria, and pinpoint missing keywords accurately.
resume:{text}
description:{job_description}
I want the response in one single string having the structure
{{"Job Description Match":"%",
"Missing Keywords":"",
"Candidate Summary":"",
"Experience":""}}
"""


def evaluate_resume(job_description, resume_text):
    # Generate content based on the input text
    output = llm.generate_content(input_prompt_template.format(text=resume_text, job_description=job_description))

    # Parse the response to extract relevant information
    response_text = output.text

    # Extracting job description match
    job_description_match = response_text.split('"Job Description Match":"')[1].split('"')[0]

    # Extracting missing keywords
    missing_keywords = response_text.split('"Missing Keywords":"')[1].split('"')[0]

    # Extracting candidate summary
    candidate_summary = response_text.split('"Candidate Summary":"')[1].split('"')[0]

    # Extracting experience
    experience = response_text.split('"Experience":"')[1].split('"')[0]

    # Return the extracted components
    return job_description_match, missing_keywords, candidate_summary, experience


# Create Gradio interface
inputs = [
    gr.Textbox(lines=10, label="Job Description"),
    gr.File(label="Upload Your Resume")
]

outputs = [
    gr.Textbox(label="Job Description Match"),
    gr.Textbox(label="Missing Keywords"),
    gr.Textbox(label="Candidate Summary"),
    gr.Textbox(label="Experience")
]

gr.Interface(
    fn=evaluate_resume,
    inputs=inputs,
    outputs=outputs,
    title="                                                                                                                  ResuMate 📑                                                                           ",
    theme="ParityError/Interstellar"
).launch(share=True)