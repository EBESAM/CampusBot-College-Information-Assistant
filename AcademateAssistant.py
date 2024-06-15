import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
#from dotenv import load_dotenv

#load_dotenv()

genai.configure(api_key="AIzaSyDuV0_YnMu0wYFLATzmmk0SdshQUQPDrhk")
os.environ["GOOGLE_API_KEY"]="AIzaSyDuV0_YnMu0wYFLATzmmk0SdshQUQPDrhk"

default_pdf_path = "C:/Users/E.B.E.S.A.M S/PycharmProjects/Myproject/SYLLABUS UPDATED.pdf"

# Initialize conversation history
conversation_history = ""

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available ", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, pdf_file=None):
    global conversation_history

    if pdf_file is None:
        pdf_path = default_pdf_path
    else:
        pdf_path = pdf_file.name

    pdf_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    get_vector_store(text_chunks)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    # Append the current question to the conversation history
    conversation_history += f"\nUser Question:\n{user_question}\n"

    # If there is a conversation history, prepend it to the user question
    if conversation_history:
        user_question_with_history = f"{conversation_history}"
    else:
        user_question_with_history = user_question

    response = chain(
        {"input_documents": docs, "question": user_question_with_history}, return_only_outputs=True
    )

    # Extract the answer from the response
    current_answer = response["output_text"]

    # Append the current answer to the conversation history
    conversation_history += f"\nBot Answer:\n{current_answer}\n"

    return conversation_history

def main():
    iface = gr.Interface(
        fn=user_input,
        inputs=["text"],
        outputs="text",  # Set live to False to disable automatic updates
        title="                                                                                                               Academate Assistant 📖📚                                                                                   ",
        description="Your Doubts about the department",
        theme="ParityError/Interstellar"
    )

    iface.launch(share=True)

if __name__ == "__main__":
    main()