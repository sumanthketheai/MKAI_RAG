# app.py
import os
import json
import shutil
import logging
from datetime import datetime
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama
from jira import JIRA
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv() #Loads .env values
# feedback_manager: this must be available in the same folder or installed as a package
# It should provide: ensure_feedback_file(), log_feedback(user, q, a, helpful, comment=None), read_feedback_stats()
from feedback_manager import log_feedback, read_feedback_stats, ensure_feedback_file
# ---------------- Logging ----------------
# ---------------- Logging ----------------
# ---------------- File-based Logging ----------------
LOG_FILE = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
# ---------------- Config (use env vars; fallback to placeholders) ----------------
# IMPORTANT: set these as environment variables in production or in a local .env loader
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://yourcompany.atlassian.net")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "you@example.com")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")  # <-- prefer env var
PROJECT_KEY = os.getenv("PROJECT_KEY", "PROJECT")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_jira_fb")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")  # or the model you want

print("JIRA URL:", os.getenv("JIRA_BASE_URL"))
print("Email:", os.getenv("JIRA_EMAIL"))

# ---------------- Ensure folders & feedback file ----------------
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs("data", exist_ok=True)
ensure_feedback_file()
# ---------------- Connect to JIRA (if credentials present) ----------------
jira = None
if JIRA_API_TOKEN:
   try:
       jira = JIRA(server=JIRA_BASE_URL, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
       logger.info("Connected to JIRA.")
   except Exception as e:
       jira = None
       logger.warning(f"Could not connect to JIRA: {e}")
else:
   logger.warning("JIRA_API_TOKEN not provided. JIRA functionality will be disabled.")
# ---------------- Chroma persistent client & collection ----------------
try:
   chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
except Exception as e:
   # Fallback to in-memory if persistent client fails (but warn)
   logger.warning(f"PersistentClient init failed: {e}. Falling back to in-memory client.")
   chroma_client = chromadb.Client()
# Setup embedding function for collection (used by Chroma collection)
embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
# Create or get collection. PersistentClient handles persistence automatically.
collection = chroma_client.get_or_create_collection(
   name="jira_tickets",
   embedding_function=embedding_fn
)
# ---------------- RCA prompt template ----------------

RCA_PROMPT = """
You are MK AI Bot — a Root Cause Analysis assistant that strictly analyzes JIRA ticket data and document context retrieved via a RAG pipeline.

Your response must be based **only** on the factual content provided in the CONTEXT section below. Do **not** infer, assume, or fabricate any information not explicitly present in the context.

---
USER QUERY:
{question}
---
CONTEXT:
{context}
---
TASK:
Generate a structured RCA summary using only the provided context. Label each section with its source ([JIRA] or [Document]).

Include the following sections if the information is available:
- Ticket ID and Summary
- Root Cause
- Impact
- Resolution
- Pending Actions or Blockers
- Recommended Next Steps for the Support Engineer

If any section lacks sufficient context, clearly state: "Not available in provided context."

Do not include general advice, assumptions, or external knowledge.
"""


PDF_PROMPT = """
You are MK AI Bot — a document summarization assistant that analyzes PDF content retrieved via a RAG pipeline.

Your response must be based **only** on the factual content provided in the CONTEXT section below. Do **not** infer, assume, or fabricate any information not explicitly present in the context.

---
USER QUERY:
{question}
---
CONTEXT:
{context}
---
TASK:
Generate a detailed summary or answer based on the user query using only the provided context. Label each section with its source ([Document]).

If any section lacks sufficient context, clearly state: "Not available in provided context."

Do not include general advice, assumptions, or external knowledge.
"""




# ---------------- Simple user DB for demo (replace with real auth) ----------------
USER_DB = {
   "sumanthkethe": {"password": "admin123", "role": "admin"},
   "pm": {"password": "pm123", "role": "project_manager"},
   "vilastajane": {"password": "vilas123", "role": "viewer"}
}
# ---------------- Streamlit UI setup ----------------
st.set_page_config(page_title="MK AI Knowledge Assistant", layout="wide")
st.markdown("<style>grammarly-extension, .grammarly-btn {display:none!important;}</style>", unsafe_allow_html=True)
if "logged_in" not in st.session_state:
   st.session_state.logged_in = False
   st.session_state.username = ""
   st.session_state.role = ""
# ---------------- Login UI ----------------
if not st.session_state.logged_in:
   st.markdown("""
<div style='text-align: center; font-size: 20px;'>
<h3>🔐 MK AI Login</h3>
</div>
   """, unsafe_allow_html=True)
   col1, col2, col3 = st.columns([1, 2, 1])
   with col2:
       username = st.text_input("Username")
       password = st.text_input("Password", type="password")
       login_clicked = st.button("Login")
       if login_clicked:
           user = USER_DB.get(username)
           if user and user["password"] == password:
               st.session_state.logged_in = True
               st.session_state.username = username
               st.session_state.role = user["role"]
               st.success(f"Welcome, {username} ({user['role']})!")
               st.rerun()
           else:
               st.error("Invalid username or password")
   st.stop()
role = st.session_state.role
# Logout button
if st.sidebar.button("Logout"):
   for key in list(st.session_state.keys()):
       del st.session_state[key]
   st.rerun()

st.sidebar.markdown("---")
# ---------------- Helper functions ----------------
def extract_pdf_text(uploaded_file) -> str:
   try:
       reader = PdfReader(uploaded_file)
       return "".join([page.extract_text() or "" for page in reader.pages])
   except Exception as e:
       logger.error(f"PDF extract error: {e}")
       return ""
def ingest_jira_tickets(jql: str):
   """Fetch JIRA issues by JQL and add to Chroma collection."""
   if jira is None:
       st.error("JIRA connection not available.")
       return []
   tickets = []
   try:
       total = jira.search_issues(jql, maxResults=0).total
       start = 0
       batch = 50
       while start < total:
           issues = jira.search_issues(jql, startAt=start, maxResults=batch)
           tickets.extend(issues)
           start += batch
   except Exception as e:
       logger.error(f"JIRA fetch error: {e}")
   docs, metas, ids = [], [], []
   for issue in tickets:
       desc = getattr(issue.fields, "description", "") or ""
       content = f"{issue.key} {issue.fields.summary}\n\n{desc}"
       docs.append(content)
       metas.append({"key": issue.key, "source_type": "jira", "source": JIRA_BASE_URL})
       ids.append(issue.key)
   if docs:
       try:
           collection.add(documents=docs, metadatas=metas, ids=ids)
           # NO collection.persist() — PersistentClient persists automatically
           logger.info(f"Added {len(docs)} JIRA docs to collection.")
       except Exception as e:
           logger.error(f"Error adding JIRA docs to collection: {e}")
   return tickets
def embed_pdfs(uploaded_files):
   """Extract text from uploaded PDFs and add to collection."""
   added = 0
   for file in uploaded_files:
       text = extract_pdf_text(file)
       if not text.strip():
           continue
       cid = f"pdf_{int(datetime.now().timestamp()*1000)}"
       try:
           collection.add(
               documents=[text],
               metadatas=[{"source": file.name, "source_type": "pdf"}],
               ids=[cid]
           )
           added += 1
       except Exception as e:
           logger.error(f"Error adding PDF {file.name} to collection: {e}")
   # NO collection.persist()
   return added
def query_rca(question: str, source_filter: str) -> str:
    where = None
    if source_filter == "JIRA only":
        where = {"source_type": "jira"}
    elif source_filter == "Documents only":
        where = {"source_type": "pdf"}

    # Check for JIRA ID in question
    jira_id = None
    tokens = question.strip().split()
    for token in tokens:
        if token.upper().startswith("JIRA-"):
            jira_id = token.upper()
            break

    if jira_id:
        try:
            res = collection.get(ids=[jira_id])
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
        except Exception as e:
            logger.error(f"Exact match failed: {e}")
            docs, metas = [], []
    else:
        try:
            res = collection.query(query_texts=[question], n_results=5, where=where)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
        except Exception as e:
            logger.error(f"Chroma query error: {e}")
            return "Error querying knowledge base."

    if not docs:
        return "No relevant context found."

    context_parts = []
    for d, m in zip(docs, metas):
        src_type = m.get("source_type", "unknown").upper()
        key = m.get("key") or m.get("source") or "unknown"
        snippet = (d[:1000] + "...") if len(d) > 1000 else d
        context_parts.append(f"[{src_type}] {key}: {snippet}")

    context = "\n\n".join(context_parts)
    if source_filter == "Documents only":
        prompt_template = PDF_PROMPT
    else:
        prompt_template = RCA_PROMPT
    prompt = prompt_template.format(context=context, question=question)

    try:
        logger.info(f"Calling LLM model: {LLM_MODEL}, Prompt length: {len(prompt)} characters")
        resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp.get("message", {}).get("content", "No response from LLM.")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return "LLM error."

# ---------------- Main UI ----------------
st.info("Hello! I'm MK AI Bot — ready to analyze JIRA tickets and documents.")
# Admin ingestion panel
if role in ["admin", "project_manager"]:
   st.sidebar.header("📥 Knowledge Ingestion")
   uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
   if uploaded and st.sidebar.button("🚀Embed PDFs"):
       with st.spinner("Embedding PDFs..."):
           count = embed_pdfs(uploaded)
           st.sidebar.success(f"✅ Embedded {count} PDFs")
   st.sidebar.markdown("### 📅 Ingest JIRA Tickets via JQL Query (Admin)")
   jql = st.sidebar.text_area("Enter JQL query")
   if st.sidebar.button("🚀 Fetch & Embed JIRA Tickets"):
       with st.spinner("Fetching from JIRA..."):
           tickets = ingest_jira_tickets(jql)
           st.sidebar.success(f"Ingested {len(tickets)} JIRA tickets")
   # Cleanup option
   
st.sidebar.markdown("---")
with st.sidebar.expander("📌 Embedded JIRA IDs", expanded=False):
    try:
        data = collection.get()
        jira_ids = [
            id_ for id_, meta in zip(data["ids"], data["metadatas"])
            if meta.get("source_type") == "jira"
        ]
        for jid in jira_ids:
            url = f"{JIRA_BASE_URL}/browse/{jid}"
            st.markdown(f"{jid}", help="View JIRA ticket in portal")
    except Exception as e:
        st.write("No JIRA IDs embedded.")
        st.error(f"Error: {e}")

with st.sidebar.expander("📎 Embedded PDFs", expanded=False):
    try:
        pdfs = [doc["source"] for doc in collection.get()["metadatas"] if doc.get("source_type") == "pdf"]
        for pdf in pdfs:
            st.markdown(f"[{pdf}](data/{pdf})", help="Click to preview or download")
    except:
        st.write("No PDFs embedded.")

st.sidebar.markdown("---")
if role in ["admin", "project_manager"]:
    if st.sidebar.button("🧹 Clear All Embeddings & Feedback Logs"):
        try:
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            os.makedirs(CHROMA_PATH, exist_ok=True)
            open("data/feedback.json", "w").write("[]")
            st.sidebar.success("✅ All embeddings and feedback logs cleared successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Cleanup failed: {e}")

# Header (different styles for admin vs viewer)
col1, col2 = st.columns([0.4, 5])
with col1:
   # if you have an image file, uncomment next line and ensure MKGold.png exists
   st.image("MKGold.png", width=60)
   pass
with col2:
   title_html = "<h1 style='color:#FFD700;font-size:28px;padding-top:5px;'>MK AI - Knowledge Bot</h1><hr>"
   if role in ["admin", "project_manager"]:
       title_html = "<h1 style='color:#FFD700;font-size:28px;padding-top:5px;'>MK AI - Knowledge Bot [Admin Console]</h1><hr>"
   st.markdown(title_html, unsafe_allow_html=True)
# Chat UI
st.markdown("### 💬 Ask a Question")
src_filter = st.radio("Source", ["All", "JIRA only", "Documents only"], horizontal=True)
question = st.text_area("Enter your question:")

with st.expander("💡 Prompt Suggestions", expanded=False):
    st.markdown("- What is the RCA for JIRA-12345?")
    st.markdown("- Provide details for JIRA-67890.")
    st.markdown("- Summarize the root cause and resolution for JIRA-54321.")
    st.markdown("- What are the pending actions for JIRA-98765?")

if st.button("Ask"):
   if question.strip():
       logger.info(f"Prompt submitted by user: {st.session_state.username}, Source Filter: {src_filter}, Question: {question}")
       with st.spinner("Analyzing..."):
           answer = query_rca(question, src_filter)
           st.markdown(f"<div style='background:#F0F8FF;padding:12px;border-radius:8px;'><strong>Answer:</strong><br>{answer}</div>", unsafe_allow_html=True)
           # Feedback buttons
           colf1, colf2 = st.columns(2)
           if colf1.button("👍 Helpful"):
               log_feedback(st.session_state.username, question, answer, True)
               st.success("Thanks for your feedback!")
           if colf2.button("👎 Not Helpful"):
               comment = st.text_input("Comment (optional)")
               if st.button("Submit feedback"):
                   log_feedback(st.session_state.username, question, answer, False, comment)
                   st.warning("Feedback recorded.")