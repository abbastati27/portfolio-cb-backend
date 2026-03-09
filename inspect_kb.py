from chromadb import PersistentClient

client = PersistentClient(path="vector_db")
collection = client.get_collection("portfolio")

data = collection.get()

for i, doc in enumerate(data["documents"]):
    print(f"\n--- DOCUMENT {i+1} ---\n")
    print(doc)