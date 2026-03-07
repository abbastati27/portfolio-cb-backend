from chromadb import PersistentClient
from litellm import completion
from sentence_transformers import SentenceTransformer

client = PersistentClient(path="vector_db")
collection = client.get_or_create_collection("portfolio")

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

LLM_MODEL = "groq/llama-3.1-8b-instant"


def retrieve_context(question):

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    return results["documents"][0]


def answer_question(question, history):

    context_chunks = retrieve_context(question)

    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": f"""
You are Abbas Tati's AI assistant.
Answer questions about Abbas using the context below.

Context:
{context}
"""
        }
    ] + history + [{"role": "user", "content": question}]

    response = completion(
        model=LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content