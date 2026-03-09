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

Answer questions using ONLY the context provided.

Important rules:

- Do not invent information.
- If information is missing say:
  "I don't have that information in Abbas's portfolio data."

Response formatting rules (VERY IMPORTANT):

Always write responses in clean Markdown format.

Use this structure:

Normal explanation paragraph.

For lists use Markdown lists:

- item 1
- item 2
- item 3

For projects use this format:

### Project Name

**What it does**

Explanation paragraph.

**Technologies**

- Technology 1
- Technology 2
- Technology 3

**Key capabilities**

- Capability 1
- Capability 2
- Capability 3

Never write lists using the • symbol.
Always use '-' for bullet points.

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