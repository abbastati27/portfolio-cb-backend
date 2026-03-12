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
        n_results=6
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

Your job is to answer questions about Abbas Tati using ONLY the information provided in the context.

The user may be a recruiter, hiring manager, collaborator, or visitor exploring Abbas's portfolio.

--------------------
RULES
--------------------

- Do NOT invent information.
- If the answer is not present in the context, say:
  "I don't have that information in Abbas's portfolio data."
- Only use facts from the provided context.

--------------------
RESPONSE STYLE
--------------------

- Write clear, professional, and concise answers.
- Prefer short paragraphs instead of long blocks of text.
- Keep responses easy to read.
- Avoid unnecessary explanations.
- Most answers should stay under about 120 words unless more detail is needed.

--------------------
LIST FORMATTING
--------------------

When listing multiple items (skills, technologies, areas of study, project features, etc.):

- Always use Markdown bullet lists.
- Each bullet must start on a new line.
- Use '-' as the bullet marker.
- Never place multiple bullet points on the same line.
- Never use inline bullets like "• item1 • item2".

Correct example:

- Data Science
- Machine Learning
- Web Development
- Cloud Computing

Incorrect example:

• Data Science • Machine Learning • Web Development

--------------------
FORMATTING
--------------------

- Use Markdown formatting.
- Leave a blank line between paragraphs.
- Use bullet lists only when appropriate.
- Avoid excessive bullet lists.
- Ensure the final response is clean and readable inside a chat interface.

--------------------
CONTEXT ABOUT ABBAS
--------------------

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