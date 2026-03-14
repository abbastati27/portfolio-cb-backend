from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from litellm import completion
import logging

# -----------------------------------
# LOGGING
# -----------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

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
    logger.info("Embedding model warmed up successfully")
except Exception as e:
    logger.error(f"Embedding warmup failed: {e}")


LLM_MODEL = "groq/llama-3.1-8b-instant"


# -----------------------------------
# RETRIEVE CONTEXT
# -----------------------------------

def retrieve_context(question):

    logger.info(f"Retrieving context for question: {question}")

    results = collection.query(
        query_texts=[question],
        n_results=6
    )

    documents = results.get("documents", [[]])[0]

    logger.info(f"Retrieved {len(documents)} documents from vector DB")

    return documents[:5]


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

RULES:
- Use ONLY the provided context.
- Do NOT invent facts.
- If information is missing say:
"I don't have that information in Abbas's portfolio data."

STYLE:
- Professional
- Clear
- Concise
- Easy to read

Use bullet lists when listing skills or technologies.

Example:

- Machine Learning
- Deep Learning
- Natural Language Processing

FORMAT:
- Use Markdown
- Leave blank lines between paragraphs
- Keep responses concise

CONTEXT:
{context}
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {"role": "user", "content": question}
    ]

    logger.info("Sending request to LLM")

    stream = completion(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=400,
        stream=True
    )

    for chunk in stream:

        token = ""

        try:
            if hasattr(chunk.choices[0], "delta"):
                token = chunk.choices[0].delta.get("content", "")
        except Exception as e:
            logger.error(f"Chunk parsing error: {e}")

        if token:
            yield token