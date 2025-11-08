# legal_rag_api.py
# Run: uvicorn legal_rag_api:app --reload
# Docs: http://127.0.0.1:8000/docs

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, io, re
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import spacy
import tiktoken
from PyPDF2 import PdfReader
import docx

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= CONFIG =================
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.0-flash"

embedder = SentenceTransformer("intfloat/e5-base-v2")
nlp = spacy.load("en_core_web_sm")
enc = tiktoken.get_encoding("cl100k_base")

client = chromadb.Client()
try:
    client.delete_collection("legal_contracts")
except:
    pass
collection = client.create_collection("legal_contracts", metadata={"hnsw:space": "cosine"})

# Store active contract in memory (for chat)
CURRENT_CONTRACT_ID = None

# ================= HELPERS =================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += "\n" + content
    return text.strip()

def extract_text_from_docx(docx_bytes: bytes) -> str:
    f = io.BytesIO(docx_bytes)
    doc = docx.Document(f)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text.strip()

def extract_text_from_txt(txt_bytes: bytes) -> str:
    return txt_bytes.decode("utf-8").strip()

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def chunk_with_overlap(text, max_tokens=150, overlap=30, tokenizer=None):
    tokens = tokenizer.encode(text)
    chunks, i = [], 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        doc = nlp(chunk_text)
        sentences = [sent.text for sent in doc.sents]
        chunk = " ".join(sentences).strip()
        chunks.append(chunk)
        i += max_tokens - overlap
    return chunks

def embed_passages(passages):
    texts = [f"passage: {t}" for t in passages]
    return embedder.encode(texts, normalize_embeddings=True)

def embed_queries(queries):
    q = [f"query: {x}" for x in queries]
    return embedder.encode(q, normalize_embeddings=True)

def add_contract_to_chromadb(contract_id, title, text):
    global CURRENT_CONTRACT_ID
    CURRENT_CONTRACT_ID = contract_id
    text = clean_text(text)
    chunks = chunk_with_overlap(text, tokenizer=enc)
    ids = [f"{contract_id}_chunk_{i}" for i in range(len(chunks))]
    metas = [{"title": title, "contract_id": contract_id} for _ in chunks]
    embs = embed_passages(chunks).tolist()
    collection.add(documents=chunks, embeddings=embs, metadatas=metas, ids=ids)
    return len(chunks)

def top_k(query, contract_id=None, k=5):
    qemb = embed_queries([query])[0].tolist()
    where_filter = {"contract_id": contract_id} if contract_id else None
    return collection.query(query_embeddings=[qemb], n_results=k, where=where_filter)

def format_context(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return "\n".join([f"From {m['title']}:\n{d}" for d, m in zip(docs, metas)])

def ask_gemini_with_context(user_query, results):
    context = format_context(results)
    prompt = f"""
You are a precise legal assistant.
Use ONLY the provided contract context to answer.
If information is missing, say "I don’t know based on the provided contract."

Question:
{user_query}

Context:
{context}

Answer concisely and accurately in formal legal language.
"""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.25,
            max_output_tokens=600
        )
    )
    return response.text.strip()

# ================= FASTAPI APP =================

app = FastAPI(title="Legal RAG Chatbot API", version="4.0")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Legal RAG Chatbot API running! Upload a contract via /chat_upload and then ask questions."}

@app.post("/chat_upload")
async def chat_upload(
    file: UploadFile = File(...),
    title: str = Form("Uploaded Contract")
):
    """
    Upload a contract file (PDF/DOCX/TXT) directly through the chat interface.
    This will replace any existing active contract context.
    """
    global CURRENT_CONTRACT_ID
    CURRENT_CONTRACT_ID = None

    content = await file.read()
    filename = file.filename.lower()

    # Choose correct parser
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(content)
    elif filename.endswith(".txt"):
        text = extract_text_from_txt(content)
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

    if not text or len(text.split()) < 20:
        return JSONResponse(status_code=400, content={"error": "File has too little readable text."})

    # Reset old context and add new contract
    client.delete_collection("legal_contracts")
    global collection
    collection = client.create_collection("legal_contracts", metadata={"hnsw:space": "cosine"})

    added_chunks = add_contract_to_chromadb(file.filename, title, text)
    CURRENT_CONTRACT_ID = file.filename

    return {"message": f"Uploaded '{file.filename}' successfully with {added_chunks} chunks.", "contract_id": CURRENT_CONTRACT_ID}


@app.post("/chat")
async def chat_with_contract(data: QueryRequest):
    """
    Chat endpoint — asks Gemini based on the most recently uploaded contract.
    """
    query = data.query.strip()
    if not query:
        return {"error": "Query cannot be empty."}

    if not CURRENT_CONTRACT_ID:
        return {"error": "No contract uploaded yet. Please upload a file first via /chat_upload."}

    results = top_k(query, contract_id=CURRENT_CONTRACT_ID, k=5)
    if not results["documents"]:
        return {"answer": "No relevant information found in the uploaded contract."}

    answer = ask_gemini_with_context(query, results)
    return {
        "query": query,
        "answer": answer,
        "contract_id": CURRENT_CONTRACT_ID,
        "retrieved_from": list({m['title'] for m in results['metadatas'][0]})
    }
