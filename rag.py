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

- Write clear and professional responses.
- Prefer short paragraphs instead of long blocks of text.
- Keep answers concise and easy to read.

Bullet formatting rules (IMPORTANT):

- Whenever listing multiple items, always use Markdown bullet lists.
- Every list item MUST start on a new line.
- Use '-' as the bullet marker.
- Never use inline bullets like "• item1 • item2".
- Never place multiple bullet items on the same line.

Correct example:

- Data Science
- Machine Learning
- Web Development
- Cloud Computing

Incorrect example:

• Data Science • Machine Learning • Web Development

Formatting:

- Use Markdown formatting.
- Use line breaks between paragraphs.
- Use bullet lists when listing skills, technologies, areas of study, or project features.

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