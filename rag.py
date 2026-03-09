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

Your job is to answer questions about Abbas Tati using ONLY the context provided below.

Important rules:

- Never invent information.
- If the answer is not in the context, say:
  "I don't have that information in Abbas's portfolio data."
- Write answers clearly and professionally.
- Assume the reader may be a recruiter, hiring manager, or collaborator.

Response formatting rules:

- Use short paragraphs.
- Use bullet points when listing skills, technologies, or features.
- Highlight project names or technologies using **bold formatting**.
- Keep answers structured and easy to read.

Project explanation format (when relevant):

Start with a short description of the project, then list:

• What the project does  
• Technologies used  
• Key capabilities or outcomes  

Context about Abbas Tati:
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