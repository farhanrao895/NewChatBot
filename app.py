
import streamlit as st
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


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from PDF files and count successes and failures."""
    text = ""
    success_count = 0  
    failure_count = 0  

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            if not pdf_reader.pages:
                st.error(f"Error reading {pdf.name}: The PDF has no pages.")
                failure_count += 1
                continue  

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            success_count += 1  

        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
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

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # Extract text and get counts
                    raw_text, success, failure = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.all_chunks = text_chunks
                    st.success(f"Processing complete.\nSuccessfully processed {success} file(s).\nFailed to process {failure} file(s).")

    for i, chat in enumerate(st.session_state.chat_history[-10:], 1):
        st.write(f"Q{i}: {chat['question']}")
        st.write(f"A{i}: {chat['answer']}")

    # Input field for new questions
    user_question = st.text_input("Ask another question")
    if st.button("Submit Question") and user_question:
        response = user_input(user_question, st.session_state.memory, st.session_state.all_chunks)
        st.session_state.chat_history.append({"question": user_question, "answer": response})
        st.write(f"Q: {user_question}")
        st.write(f"A: {response}")

if __name__ == "__main__":
    main()
