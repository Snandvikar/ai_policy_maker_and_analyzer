import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

from pypdf import PdfReader
import docx

# Config
DATA_DIR = "data/policy_reports"
VECTOR_DB_DIR = "vector_db_store"
COLLECTION_NAME = "india_policy_reports"

# Load embedding model (lightweight + good quality)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma
client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory=VECTOR_DB_DIR
    )
)

collection = client.get_or_create_collection(name=COLLECTION_NAME)


# ─────────────────────────────────────────────
# File Readers
# ─────────────────────────────────────────────

def read_pdf(path):
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    return text


def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

import re

def contextual_chunk(text, chunk_size=800):

    # Split by headings (common in govt reports)
    sections = re.split(r'\n[A-Z][A-Z\s]{5,}\n', text)

    chunks = []

    for section in sections:
        
        # further split large sections
        if len(section) > chunk_size:
            
            paragraphs = section.split("\n\n")

            current = ""
            for para in paragraphs:
                
                if len(current) + len(para) < chunk_size:
                    current += "\n\n" + para
                else:
                    chunks.append(current.strip())
                    current = para

            if current:
                chunks.append(current.strip())

        else:
            chunks.append(section.strip())

    return [c for c in chunks if len(c) > 100]


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────

def ingest():

    data_path = Path(DATA_DIR)

    doc_id = 0

    for file in data_path.glob("*"):

        print(f"Ingesting {file}")

        if file.suffix == ".pdf":
            text = read_pdf(file)

        elif file.suffix == ".docx":
            text = read_docx(file)

        elif file.suffix == ".txt":
            text = read_txt(file)

        else:
            continue

        chunks = contextual_chunk(text)

        embeddings = model.encode(chunks)

        ids = [f"{file.stem}_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=[{"source": file.name}] * len(chunks)
        )

    # client.persist()

    print("Policy documents stored in vector DB")


if __name__ == "__main__":
    ingest()