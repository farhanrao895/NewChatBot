import streamlit as st
import io as io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from dotenv import load_dotenv
import requests
import base64

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Azure AD credentials
tenant_id = "73b32ac8-173b-4ddc-ad2e-acba48de9843"
client_id = "c39cfc40-36ca-465e-bc63-fb8de4372ff9"
client_secret = "Mgg8Q~zfDp6y-CKJ4Nqta75n1z6-A4PYe9SwLa5d"
# sharing_url=""
# sharing_url = "https://freyssinetsa-my.sharepoint.com/:f:/g/personal/joel_a_fsa_com_sa/Ei0JHI3cQ-9AnWNT_raORLwBspJ513ffBE3H8h4x2t5fsA?e=2PxUcO"

def get_access_token():
    auth_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default"
    }
    response = requests.post(auth_url, data=payload)
    response.raise_for_status()
    return response.json().get("access_token")

def get_graph_share_id(sharing_url):
    print('sharingurl',sharing_url)
    encoded_url = base64.urlsafe_b64encode(sharing_url.encode("utf-8")).decode("utf-8").strip("=")
    return f"u!{encoded_url}"

def fetch_shared_files(access_token, graph_share_id):
    url = f"https://graph.microsoft.com/v1.0/shares/{graph_share_id}/driveItem/children"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    files = response.json().get("value", [])
    return files

def download_file(file_url, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(file_url, headers=headers)
    response.raise_for_status()
    return response.content

def process_pdf_files(files, access_token):
    """Fetch and process PDF files from SharePoint."""
    text = ""
    success_count = 0
    failure_count = 0

    for file in files:
        if "file" in file and file["name"].endswith(".pdf"):
            file_url = file["@microsoft.graph.downloadUrl"]
            try:
                # Download the PDF content
                pdf_content = download_file(file_url, access_token)

                # Wrap the bytes content in a BytesIO object
                pdf_file = io.BytesIO(pdf_content)

                # Read the PDF using PdfReader
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                success_count += 1
            except Exception as e:
                st.error(f"Error processing file {file['name']}: {e}")
                failure_count += 1

    return text, success_count, failure_count


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=12500, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Define a conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, respond with "Answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def hybrid_search(user_question, vector_store):
    """Perform hybrid search using vector similarity and keyword matching."""
    # Perform vector-based similarity search and get top 300 chunks
    vector_results = vector_store.similarity_search(user_question, k=300)

    # Perform keyword-based search over the vector_results
    keyword_results = []
    for chunk in vector_results:
        if any(word.lower() in chunk.page_content.lower() for word in user_question.split()):
            keyword_results.append(chunk)

    combined_results = []
    seen_chunks = set()

    # Add vector-based results first (priority)
    for chunk in vector_results:
        content = chunk.page_content
        if content not in seen_chunks:
            combined_results.append(chunk)
            seen_chunks.add(content)

    # Add keyword-based results
    for chunk in keyword_results:
        content = chunk.page_content
        if content not in seen_chunks:
            combined_results.append(chunk)
            seen_chunks.add(content)

    combined_results = combined_results[:300]

    return combined_results

def user_input(user_question, memory, all_chunks):
    """Handle user input and generate a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform hybrid search
    docs = hybrid_search(user_question, vector_store)

    # Prepare context by joining unique page content
    context = "\n".join(doc.page_content for doc in docs)

    # Get conversational chain and generate response
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "context": context, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with Multiple PDFs using Gemini 1.5 Pro")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=10)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []

  # Initialize or retrieve sharing_url in session state
    if "sharing_url" not in st.session_state:
        st.session_state.sharing_url = ""
     
  # Sidebar to fetch files from SharePoint
    with st.sidebar:
        st.title("Menu:")
        st.session_state.sharing_url = st.text_input(
            "Enter Sharepoint Sharing URL:", 
            value=st.session_state.sharing_url
        )
        if st.button("Fetch and Process Files from SharePoint"):
            with st.spinner("Fetching files from SharePoint..."):
                try:
                    access_token = get_access_token()
                    graph_share_id = get_graph_share_id(st.session_state.sharing_url)
                    files = fetch_shared_files(access_token, graph_share_id)
                    raw_text, success, failure = process_pdf_files(files, access_token)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.all_chunks = text_chunks
                    st.success(f"Successfully processed {success} file(s), failed {failure} file(s).")
                except Exception as e:
                    st.error(f"Error fetching files: {e}")

    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history[-10:]:
        st.write(f"Q: {chat['question']}")
        st.write(f"A: {chat['answer']}")

    # Input field for new questions
    user_question = st.text_input("Ask another question")
    if st.button("Submit Question") and user_question:
        response = user_input(user_question, st.session_state.memory, st.session_state.all_chunks)
        st.session_state.chat_history.append({"question": user_question, "answer": response})
        st.write(f"Q: {user_question}")
        st.write(f"A: {response}")

if __name__ == "__main__":
    main()
