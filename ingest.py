from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

client = PersistentClient(path="vector_db")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="portfolio",
    embedding_function=embedding_function
)

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

collection.add(
    ids=ids,
    documents=documents
)

print("Knowledge base indexed successfully.")