import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Environment Variables Load Karein
load_dotenv()

# --- Page Config & Theme ---
st.set_page_config(page_title="Medi-Query AI", page_icon="⚕️", layout="centered")

# Medical Blue/Teal Theme for better UI
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-title { color: #0891b2; font-size: 38px; font-weight: bold; text-align: center; }
    .disclaimer { background-color: #fee2e2; padding: 12px; border-radius: 8px; color: #991b1b; font-size: 13px; border: 1px solid #fecaca; margin-bottom: 20px; }
    .stChatMessage { background-color: #ffffff !important; border: 1px solid #e2e8f0; border-radius: 10px; color: #1e293b !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>⚕️ Medi-Query AI Research</div>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>⚠️ <b>DISCLAIMER:</b> This is an AI research tool based on your uploaded medical papers. Consult a doctor for clinical decisions.</div>", unsafe_allow_html=True)

# --- Auto-Ingestion Logic (Main Chiz) ---
@st.cache_resource
def load_medical_db():
    DB_PATH = "./chroma_db_medical"
    DATA_PATH = "medical_data/"
    
    # Check karein agar database folder nahi hai
    if not os.path.exists(DB_PATH):
        # Check karein agar PDFs folder hai aur usme files hain
        if os.path.exists(DATA_PATH) and len(os.listdir(DATA_PATH)) > 0:
            with st.spinner("🚀 Database missing on Cloud! Ingesting medical papers... Please wait."):
                from ingestion import create_medical_db
                create_medical_db()
        else:
            st.error("⚠️ 'medical_data' folder is empty. Please push PDFs to GitHub first!")
            st.stop()
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Initialize DB
db = load_medical_db()

# --- RAG Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 5})

template = """You are a professional Medical Research Assistant powered by the GPT-4o-mini model. 

INSTRUCTION HIERARCHY:
1. MEDICAL DOCUMENTS (Priority): For any medical or clinical questions, first search the provided context below. If the answer is in the files, use it as your primary source and provide citations.
2. IDENTITY: If asked about your model or name, identify yourself as GPT-4o-mini, a specialized Medical RAG Assistant.
3. GENERAL KNOWLEDGE: If the question is NOT about the medical files (e.g., general history, geography, or daily life) OR if the medical answer isn't in the files, use your own internal knowledge to provide a helpful answer. Do NOT say "I don't know" or "Not in context" for non-medical topics.

Context: {context}

Question: {question}

Analysis/Answer:"""


prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"Reference: {doc.metadata.get('source')}\n{doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# --- Chat Interface ---
if "med_history" not in st.session_state:
    st.session_state.med_history = []

for msg in st.session_state.med_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about WHO PEN, Stroke Toolkit, or Diabetes..."):
    st.session_state.med_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing medical literature..."):
            response = rag_chain.invoke(query)
            st.markdown(response)
            
            # Show Citations
            source_docs = retriever.invoke(query)
            with st.expander("📚 Verified Medical Sources"):
                sources = {doc.metadata.get('source') for doc in source_docs}
                for s in sources:
                    st.write(f"📍 {s}")
            
            st.session_state.med_history.append({"role": "assistant", "content": response})