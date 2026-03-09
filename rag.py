from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from litellm import completion

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
# MODEL WARMUP
# -----------------------------------
# Forces embedding model download & load during server start
# Prevents 1-minute delay on first chatbot message

try:
    embedding_function(["warmup"])
except Exception:
    pass


LLM_MODEL = "groq/llama-3.1-8b-instant"


# -----------------------------------
# RETRIEVE CONTEXT
# -----------------------------------

def retrieve_context(question):

    results = collection.query(
        query_texts=[question],
        n_results=8
    )

    documents = results.get("documents", [[]])[0]

    return documents


# -----------------------------------
# ANSWER QUESTION
# -----------------------------------

def answer_question(question, history):

    context_chunks = retrieve_context(question)

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = f"""
You are Abbas Tati's AI portfolio assistant.

Your job is to answer questions about Abbas Tati using ONLY the provided context.

Rules:
- Do not invent information.
- If the answer is not in the context, say you don't have that information.
- Keep answers clear and concise.
- If a recruiter asks about skills, experience, or projects, explain them professionally.
- If someone asks how to contact Abbas, provide his contact information if available.

Context about Abbas:
{context}
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {"role": "user", "content": question}
    ]

    response = completion(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content