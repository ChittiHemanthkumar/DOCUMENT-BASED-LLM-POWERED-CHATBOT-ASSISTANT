"""import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
import requests
import pdfkit
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.title("Web Content Question Answering")

# Sidebar for URL input
st.sidebar.title("Enter URLs")
urls = st.sidebar.text_area("Paste URLs, separated by commas").split(',')

# Function to fetch, convert to PDF, and process text
@st.cache(allow_output_mutation=True, show_spinner=True)
def fetch_and_process_content(urls, question):
    raw_text = ''

    # Fetch and convert HTML to PDF
    for url in urls:
        try:
            response = requests.get(url.strip())
            if response.status_code == 200:
                # Convert HTML to PDF in memory
                pdf_bytes = pdfkit.from_string(response.text, False)
                pdf_file = BytesIO(pdf_bytes)

                # Extract text from PDF
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        raw_text += content
        except Exception as e:
            st.error(f"Failed to process URL {url}: {e}")
            continue

    if not raw_text:
        st.error("No text could be extracted from the URLs provided.")
        return None

    # Split text into chunks and create embeddings
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    # Load question answering chain and run the question
    chain = load_qa_chain(OpenAI())
    docs = document_search.similarity_search(question, top_k=3)
    result = chain.run(input_documents=docs, question=question)

    return result

# Main content
if urls and any(url.strip() for url in urls):
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        with st.spinner('Fetching and processing content...'):
            result = fetch_and_process_content(urls, question)
            if result:
                st.write("### Answer:")
                st.write(result)
else:
    st.write("Please enter URLs in the sidebar.")""" 

import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
import requests
import pdfkit
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the title of the app
st.title("Web Content Question Answering")

# Sidebar for URL input
st.sidebar.title("Enter URLs")
urls = st.sidebar.text_area("Paste URLs, separated by commas").split(',')

# Correct path to wkhtmltopdf
path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'  # Update this path as necessary
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Define the function to fetch, convert, and process content
@st.experimental_memo(show_spinner=True)
def fetch_and_process_content(urls, question):
    raw_text = ''
    errors = []

    for url in urls:
        try:
            response = requests.get(url.strip())
            if response.status_code == 200:
                pdf_bytes = pdfkit.from_string(response.text, False, configuration=config)
                pdf_file = BytesIO(pdf_bytes)
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        raw_text += content
        except Exception as e:
            errors.append(f"Failed to process URL {url.strip()}: {e}")
            continue

    if not raw_text:
        errors.append("No text could be extracted from the URLs provided.")
        return None, errors

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI())
    docs = document_search.similarity_search(question, top_k=3)
    result = chain.run(input_documents=docs, question=question)

    return result, errors

# Main content display
if urls and any(url.strip() for url in urls):
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        with st.spinner('Fetching and processing content...'):
            result, errors = fetch_and_process_content(urls, question)
            if errors:
                for error in errors:
                    st.error(error)
            if result:
                st.write("### Answer:")
                st.write(result)
else:
    st.write("Please enter URLs in the sidebar.")
