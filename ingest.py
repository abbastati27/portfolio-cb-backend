from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# -----------------------------------
# CONFIG
# -----------------------------------

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# -----------------------------------
# CHROMA SETUP
# -----------------------------------

client = PersistentClient(path="vector_db")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="portfolio",
    embedding_function=embedding_function
)

# -----------------------------------
# CHUNKING FUNCTION
# -----------------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------------
# LOAD & CHUNK DOCUMENTS
# -----------------------------------

documents = []
ids = []
metadatas = []

doc_counter = 0

for file in Path("knowledge-base").glob("*.md"):

    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        continue

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):

        documents.append(chunk)

        ids.append(f"{file.stem}_{i}")

        metadatas.append({
            "source": file.name
        })

        doc_counter += 1

print(f"Created {doc_counter} chunks from knowledge base")

# -----------------------------------
# STORE IN CHROMA
# -----------------------------------

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("Knowledge base indexed successfully.")