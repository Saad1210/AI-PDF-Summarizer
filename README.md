# AI-PDF-Summarizer
Project Overview

This project involves developing an AI-based system to automatically summarize content from PDF documents. The system leverages natural language processing (NLP) techniques to extract key points and generate concise summaries, making it easier to digest large amounts of information quickly.

## Features
- PDF Parsing: Converts PDF content into text for analysis.
- Text Summarization: Uses Google Gemini Pro API to generate high-quality summaries of the extracted text.
- Efficient Retrieval: Employs FAISS for fast and accurate information retrieval from large datasets.
- Batch Processing: Handles multiple PDF documents simultaneously.

## Tech Stack
Programming Language : Python 3.10

### Libraries
- PyPDF2 : For extracting text from PDF documents.
- Google Gemini Pro API: For advanced NLP and summarization capabilities. 
- Langchain : For building and managing language models and chains.
- FAISS: For efficient similarity search and clustering in large datasets.
- Streamlit : For creating a web-based interface to interact with the summarization tool.
