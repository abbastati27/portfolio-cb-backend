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

    # Use top 5 chunks to reduce noise
    return documents[:5]


# -----------------------------------
# ANSWER QUESTION
# -----------------------------------

def answer_question(question, history):

    context_chunks = retrieve_context(question)

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = f"""
You are Abbas Tati's AI portfolio assistant.

Answer questions about Abbas Tati using ONLY the context provided below.

Rules:

- Do not invent information.
- If the answer is not present in the context, say:
  "I don't have that information in Abbas's portfolio data."
- Assume the user may be a recruiter, hiring manager, or collaborator.

Response style:

- Write clear, professional responses.
- Prefer short paragraphs instead of long blocks of text.
- Use bullet points only when listing items like skills, technologies, areas of study, or capabilities.
- Do not overuse bullet points.
- Keep answers concise and easy to read.
- Avoid repeating information unnecessarily.

Formatting:

- Use clean Markdown formatting.
- Use line breaks between paragraphs.
- Use bullet lists only when helpful.

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
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content