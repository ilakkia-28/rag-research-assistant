import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2702/2702154.png", width=100)
    st.title("📄 AI Research Assistant")
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Upload a PDF file")
    st.markdown("2. Ask any question")
    st.markdown("3. Get instant AI answers!")
    st.markdown("---")
    st.markdown("Built with LangChain + Groq 🚀")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []

# Main area
st.title("🤖 AI Research Assistant")
st.markdown("##### Upload any PDF and ask questions — powered by LLaMA AI")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    with col1:
        st.success("✅ PDF Uploaded: " + uploaded_file.name)
    with col2:
        st.info(f"📚 Pages: {len(pdf_reader.pages)} | Chunks: {len(chunks)}")

    st.markdown("---")

    # Show chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # Chat input
    question = st.chat_input("💬 Ask a question about your PDF...")

    if question:
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🤔 Thinking..."):
            context = "\n\n".join(chunks[:5])
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=GROQ_API_KEY
            )
            prompt = f"Using this context:\n{context}\n\nAnswer this question: {question}"
            response = llm.invoke(prompt)

        answer = response.content
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})