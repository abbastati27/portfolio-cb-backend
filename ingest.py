from pathlib import Path
from openai import OpenAI
from chromadb import PersistentClient

client = PersistentClient(path="vector_db")
collection = client.get_or_create_collection("portfolio")

openai = OpenAI()

EMBED_MODEL = "text-embedding-3-small"

documents = []
ids = []

for i, file in enumerate(Path("knowledge-base").glob("*.md")):

    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        continue

    documents.append(text)
    ids.append(str(i))

print(f"Loaded {len(documents)} documents")

embeddings = openai.embeddings.create(
    model=EMBED_MODEL,
    input=documents
)

vectors = [e.embedding for e in embeddings.data]

collection.add(
    ids=ids,
    documents=documents,
    embeddings=vectors
)

print("Knowledge base indexed successfully.")