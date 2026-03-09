from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# -----------------------------------
# CONFIG
# -----------------------------------

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# -----------------------------------
# CHROMA SETUP
# -----------------------------------

client = PersistentClient(path="vector_db")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Delete old collection if it exists
try:
    client.delete_collection("portfolio")
except:
    pass

collection = client.create_collection(
    name="portfolio",
    embedding_function=embedding_function
)

# -----------------------------------
# SMART CHUNKING
# -----------------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

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
            "source": file.name,
            "chunk": i
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