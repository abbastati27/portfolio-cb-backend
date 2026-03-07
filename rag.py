from chromadb import PersistentClient
from openai import OpenAI
from litellm import completion

client = PersistentClient(path="vector_db")
collection = client.get_or_create_collection("portfolio")

openai = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "openai/gpt-4.1-mini"


def retrieve_context(question):

    query_embedding = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[question]
    ).data[0].embedding

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