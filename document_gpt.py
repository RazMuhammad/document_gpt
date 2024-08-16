import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
import requests
import os  # import os to get API key from environment variables

# Initialize dotenv to load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the model and Pinecone
embedder = SentenceTransformer("evilfreelancer/enbeddrus-v0.2")

# Get API keys from Streamlit secrets
pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone_environment = st.secrets["pinecone_environment"]
pinecone_index_name = st.secrets["pinecone_index_name"]
claude_api_key = st.secrets["claude_api_key"]

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index = pinecone.Index(pinecone_index_name)

# Claude AI API endpoint
claude_api_url = "https://api.anthropic.com/v1/complete"  # Ensure this is the correct endpoint

# Streamlit UI for document upload and query input
st.title("Generative AI Document Search")
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Read and decode the document
    document_text = uploaded_file.read().decode('utf-8')
    st.write("Document content:", document_text)

    # Embed the document text
    doc_embedding = embedder.encode(document_text)

    # Store the embedding in Pinecone
    doc_id = "document_1"  # Unique ID for the document
    index.upsert([(doc_id, doc_embedding.tolist())])

    st.write("Document uploaded and embedded successfully.")

# User input for query
query = st.text_input("Enter your query")

if query:
    # Embed the query
    query_embedding = embedder.encode(query)

    # Search in Pinecone
    results = index.query(query_embedding.tolist(), top_k=3)

    # Retrieve the top matching document chunks
    matching_chunks = [result['values'] for result in results['matches']]

    # Prepare the prompt
    prompt = f"Based on the document: {matching_chunks} \nAnswer the query: {query}"

    # Generate response using Claude AI
    headers = {
        "Authorization": f"Bearer {claude_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "claude-v1",  # Replace with the correct model name if different
        "prompt": prompt,
        "max_tokens": 200
    }

    response = requests.post(claude_api_url, headers=headers, json=data)
    response_text = response.json().get('completion', '')

    st.write("Response:", response_text)
