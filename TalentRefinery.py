import gradio as gr
import google.generativeai as genai
import os
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
As an experienced recruitment expert
with profound knowledge in technology, software engineering, data science, 
and big data engineering, your role involves evaluating resumes against job descriptions.
Recognizing the competitive job market, provide top-notch assistance for resume improvement.
Your goal is to evaluate the resume against the given job description and write pros, cons, and your personal reflection, 
and based on the info who seems most suitable for the job and assign a percentage match based on key criteria and explain your choice in a few sentences.
resume:{text}
description:{job_description}
I want the response in one single string having the structure
{{"Pros":"",
"Cons":"",
"Summary":"",
"Matching Percentage":""}}
"""


def evaluate_resume(job_description, resume1_text, resume2_text):
    # Generate content based on the input text for resume 1
    output1 = llm.generate_content(input_prompt_template.format(text=resume1_text, job_description=job_description))

    # Parse the response to extract relevant information for resume 1
    response_text1 = output1.text
    Pros_of_the_first_candidate = response_text1.split('"Pros":"')[1].split('"')[0]
    Cons_of_the_first_candidate = response_text1.split('"Cons":"')[1].split('"')[0]
    Summary_of_the_first_candidate = response_text1.split('"Summary":"')[1].split('"')[0]
    Matching_percentage_of_the_first_candidate = response_text1.split('"Matching Percentage":"')[1].split('"')[0]

    # Generate content based on the input text for resume 2
    output2 = llm.generate_content(input_prompt_template.format(text=resume2_text, job_description=job_description))

    # Parse the response to extract relevant information for resume 2
    response_text2 = output2.text
    Pros_of_the_Second_candidate = response_text2.split('"Pros":"')[1].split('"')[0]
    Cons_of_the_Second_candidate = response_text2.split('"Cons":"')[1].split('"')[0]
    Summary_of_the_Second_candidate = response_text2.split('"Summary":"')[1].split('"')[0]
    Matching_percentage_of_the_Second_candidate= response_text2.split('"Matching Percentage":"')[1].split('"')[0]

    # Return the extracted components for each resume separately
    return Pros_of_the_first_candidate , Cons_of_the_first_candidate, Summary_of_the_first_candidate, Matching_percentage_of_the_first_candidate, Pros_of_the_Second_candidate, Cons_of_the_Second_candidate, Summary_of_the_Second_candidate, Matching_percentage_of_the_Second_candidate

# Create Gradio interface
inputs = [
    gr.Textbox(lines=10, label="Job Description"),
    gr.File(label="Upload First Resume"),
    gr.File(label="Upload Second Resume")
]

outputs = [
    gr.Textbox(label="Pros of the first candidate"),
    gr.Textbox(label=" Cons of the first candidate"),
    gr.Textbox(label="Summary of the first candidate"),
    gr.Textbox(label="Matching percentage of the first candidate"),
    gr.Textbox(label="Pros of the Second candidate"),
    gr.Textbox(label="Cons of the Second candidate"),
    gr.Textbox(label="Summary of the Second candidate"),
    gr.Textbox(label="Matching percentage of the Second candidate")
]

gr.Interface(
    fn=evaluate_resume,
    inputs=inputs,
    outputs=outputs,
    title="                                                                                                                 Candidate evaluation system - Enhance Your Resume                                                                                                       ",
    theme="ParityError/Interstellar"
).launch()
import gradio as gr
import google.generativeai as genai
import os
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
As an experienced recruitment expert
with profound knowledge in technology, software engineering, data science, 
and big data engineering, your role involves evaluating resumes against job descriptions.
Recognizing the competitive job market, provide top-notch assistance for resume improvement.
Your goal is to evaluate the resume against the given job description and write pros, cons, and your personal reflection, 
and based on the info who seems most suitable for the job and assign a percentage match based on key criteria and explain your choice in a few sentences.
resume:{text}
description:{job_description}
I want the response in one single string having the structure
{{"Pros":"",
"Cons":"",
"Summary":"",
"Matching Percentage":""}}
"""


def evaluate_resume(job_description, resume1_text, resume2_text):
    # Generate content based on the input text for resume 1
    output1 = llm.generate_content(input_prompt_template.format(text=resume1_text, job_description=job_description))

    # Parse the response to extract relevant information for resume 1
    response_text1 = output1.text
    Pros_of_the_first_candidate = response_text1.split('"Pros":"')[1].split('"')[0]
    Cons_of_the_first_candidate = response_text1.split('"Cons":"')[1].split('"')[0]
    Summary_of_the_first_candidate = response_text1.split('"Summary":"')[1].split('"')[0]
    Matching_percentage_of_the_first_candidate = response_text1.split('"Matching Percentage":"')[1].split('"')[0]

    # Generate content based on the input text for resume 2
    output2 = llm.generate_content(input_prompt_template.format(text=resume2_text, job_description=job_description))

    # Parse the response to extract relevant information for resume 2
    response_text2 = output2.text
    Pros_of_the_Second_candidate = response_text2.split('"Pros":"')[1].split('"')[0]
    Cons_of_the_Second_candidate = response_text2.split('"Cons":"')[1].split('"')[0]
    Summary_of_the_Second_candidate = response_text2.split('"Summary":"')[1].split('"')[0]
    Matching_percentage_of_the_Second_candidate= response_text2.split('"Matching Percentage":"')[1].split('"')[0]

    # Return the extracted components for each resume separately
    return Pros_of_the_first_candidate , Cons_of_the_first_candidate, Summary_of_the_first_candidate, Matching_percentage_of_the_first_candidate, Pros_of_the_Second_candidate, Cons_of_the_Second_candidate, Summary_of_the_Second_candidate, Matching_percentage_of_the_Second_candidate

# Create Gradio interface
inputs = [
    gr.Textbox(lines=10, label="Job Description"),
    gr.File(label="Upload First Resume"),
    gr.File(label="Upload Second Resume")
]

outputs = [
    gr.Textbox(label="Pros of the first candidate"),
    gr.Textbox(label=" Cons of the first candidate"),
    gr.Textbox(label="Summary of the first candidate"),
    gr.Textbox(label="Matching percentage of the first candidate"),
    gr.Textbox(label="Pros of the Second candidate"),
    gr.Textbox(label="Cons of the Second candidate"),
    gr.Textbox(label="Summary of the Second candidate"),
    gr.Textbox(label="Matching percentage of the Second candidate")
]

gr.Interface(
    fn=evaluate_resume,
    inputs=inputs,
    outputs=outputs,
    title="                                                                                                                Talent Refinery Model 🧠                                                                                                   ",
    theme="ParityError/Interstellar"
).launch(share=True)