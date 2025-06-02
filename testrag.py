# # # File: RAG/test_rag.py

# import os
# from pinecone import Pinecone as PineconeClient
# from langchain_openai import OpenAIEmbeddings   # updated import path
# # If you still have langchain<0.2, you could use:
# #   from langchain.embeddings.openai import OpenAIEmbeddings

# # ──────────────────────────────────────────────────────────
# # 1) Configure your keys (either hard‐code or export them)
# # ──────────────────────────────────────────────────────────
# PINECONE_API_KEY = "pcsk_6esMas_Tfb3tkmUXKj51nQX7geiCV9s4oEyxe5q8w9EVx8EMpG9tWAkk3e98fYUbTv3RV3"
# PINECONE_ENV     = "us-east-1"
# OPENAI_API_KEY   = "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"

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# # ──────────────────────────────────────────────────────────
# # 2) Initialize Pinecone client (v7 style) and grab your index
# # ──────────────────────────────────────────────────────────
# pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# #  ↑ This is the only “Pinecone” import you need. Don’t call pinecone.init()

# # Now get a handle to “energy-rag” exactly as ingest_interactions does:
# index = pc.Index("energy-rag")   # ← correct: pc.Index, not pinecone.Index

# # ──────────────────────────────────────────────────────────
# # 3) Initialize the same OpenAIEmbeddings you used to ingest
# # ──────────────────────────────────────────────────────────
# embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# def get_relevant_examples(user_query: str, top_k: int = 5):
#     """
#     1) Embed the query
#     2) Query Pinecone for the top_k matches
#     3) Return a list of (prompt, response) pairs
#     """
#     # 1) Embed
#     q_vec = embedder.embed_query(user_query)

#     # 2) Pinecone vector search
#     results = index.query(
#         vector=q_vec,
#         top_k=top_k,
#         include_metadata=True
#     )

#     # 3) Extract prompt/response from metadata
#     examples = []
#     for match in results.matches:
#         md = match.metadata
#         p = md.get("prompt", "")
#         r = md.get("response", "")
#         examples.append((p, r))
#     return examples

# if __name__ == "__main__":
#     # 4) Test it with a known question
#     test_question = "How many ISO’s are there?"

#     examples = get_relevant_examples(test_question, top_k=3)
#     print(f"\nRAG results for question: {test_question}\n")
#     for i, (p, r) in enumerate(examples, start=1):
#         print(f"--- Example {i} ---")
#         print("Prompt:  ", p)
#         print("Response:", r)
#         print()

import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings   # updated import path

# ──────────────────────────────────────────────────────────
# 1) Configure your keys (either hard-code or export them)
# ──────────────────────────────────────────────────────────
PINECONE_API_KEY = "pcsk_6esMas_Tfb3tkmUXKj51nQX7geiCV9s4oEyxe5q8w9EVx8EMpG9tWAkk3e98fYUbTv3RV3"
PINECONE_ENV     = "us-east-1"
OPENAI_API_KEY   = "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"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ──────────────────────────────────────────────────────────
# 2) Initialize Pinecone client (v6.0.2) and grab your index
# ──────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index("energy-rag")   # assume "energy-rag" already exists

# ──────────────────────────────────────────────────────────
# 3) Initialize the same OpenAIEmbeddings you used to ingest
# ──────────────────────────────────────────────────────────
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

def get_relevant_examples(user_query: str, top_k: int = 5):
    """
    1) Embed the query
    2) Query Pinecone for the top_k matches
    3) Return a list of (prompt, response) pairs
    """
    # 1) Embed
    q_vec = embedder.embed_query(user_query)

    # 2) Pinecone vector search
    results = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True
    )

    # 3) Extract prompt/response from metadata
    examples = []
    for match in results.matches:
        md = match.metadata
        p = md.get("original_prompt", "")
        r = md.get("original_response", "")
        examples.append((p, r))
    return examples

if __name__ == "__main__":
    # 4) Test it with a known question
    test_question = "How many ISO’s are there?"

    examples = get_relevant_examples(test_question, top_k=3)
    print(f"\nRAG results for question: {test_question}\n")
    for i, (p, r) in enumerate(examples, start=1):
        print(f"--- Example {i} ---")
        print("Prompt:   ", p)
        print("Response: ", r)
        print()

