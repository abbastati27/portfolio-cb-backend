from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from litellm import completion

client = PersistentClient(path="vector_db")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="portfolio",
    embedding_function=embedding_function
)

LLM_MODEL = "groq/llama-3.1-8b-instant"


def retrieve_context(question):

    results = collection.query(
        query_texts=[question],
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