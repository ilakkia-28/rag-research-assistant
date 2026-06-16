import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Research Assistant", page_icon="📄")
st.title("📄 AI Research Assistant")
st.write("Upload a PDF and ask questions from it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("✅ PDF Uploaded!")

    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    st.info(f"📚 Split into {len(chunks)} chunks!")
    st.success("🧠 AI has read your PDF!")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        context = "\n\n".join(chunks[:5])

        llm = ChatGroq(
            model= "llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        )
        prompt = f"Using this context:\n{context}\n\nAnswer this question: {question}"
        response = llm.invoke(prompt)

        st.write("### 🤖 Answer:")
        st.write(response.content)