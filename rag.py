from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from litellm import completion
import re

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

    return documents[:5]


# -----------------------------------
# CLEAN OUTPUT (bullet + formatting fix)
# -----------------------------------

def clean_output(text):

    lines = text.split("\n")

    processed = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            processed.append("")
            continue

        # convert plain list items into markdown bullets
        if (
            not stripped.startswith("-")
            and not stripped.startswith("#")
            and len(stripped.split()) <= 6
            and stripped[0].isupper()
        ):
            processed.append(f"- {stripped}")
        else:
            processed.append(stripped)

    return "\n".join(processed)


# -----------------------------------
# STREAM ANSWER
# -----------------------------------

def answer_question(question, history):

    context_chunks = retrieve_context(question)

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = f"""
You are Abbas Tati's AI portfolio assistant.

You help visitors understand Abbas's experience, projects, skills, and background.

Users may include recruiters, hiring managers, collaborators, or people exploring his portfolio.

--------------------------------
RULES
--------------------------------

- Use ONLY the provided context.
- Do NOT invent facts.
- If information is missing say:
"I don't have that information in Abbas's portfolio data."

--------------------------------
STYLE
--------------------------------

Write responses that are:

- Professional
- Clear
- Concise
- Easy to read in a chat interface

Prefer:

Short paragraphs.

--------------------------------
LISTS
--------------------------------

When listing items like skills or technologies ALWAYS format them as Markdown bullet lists.

Example:

- Machine Learning
- Deep Learning
- Natural Language Processing

Rules:

- Every bullet must start with "-"
- Each bullet must appear on its own line
- Never place multiple bullet items on one line
- Never use inline bullets like "• item • item"

--------------------------------
FORMATTING
--------------------------------

- Use Markdown
- Leave a blank line between paragraphs
- Keep responses concise (usually under 120 words)

--------------------------------
CONTEXT ABOUT ABBAS
--------------------------------

{context}
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {"role": "user", "content": question}
    ]

    stream = completion(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=400,
        stream=True
    )

    for chunk in stream:

        token = chunk.choices[0].delta.content or ""

        yield token