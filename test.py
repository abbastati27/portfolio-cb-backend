from chromadb import PersistentClient

client = PersistentClient(path="vector_db")
collection = client.get_collection("portfolio")

print(collection.count())