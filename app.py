import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.create_documents([text])
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    prompt_template = """Answere the question as precise and meaningful as possible from the given context.
                        if answer is not in provided context return 'Answere is not available in the given context'
                        dont provide wrong answeres\n
                        Context : \n {context}
                        Questions : \n {question} \n
                        Answere : """
    
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents" : docs,
                      "question" : user_question}, return_only_outputs = True)
    
    print(response)

    st.write("Reply : ", response["output_text"])

def main():
    st.set_page_config("Chat With Your PDF")
    st.header("Chat with Your PDF 💁")

    user_question = st.text_input("Ask a Question Related to the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu : ")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()