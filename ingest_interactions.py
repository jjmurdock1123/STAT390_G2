# #!/usr/bin/env python3
# """
# ingest_interactions.py

# Interactively ask for prompt/response pairs and upsert them into a Pinecone index.
# This version uses Pinecone v7’s Pinecone class (no pinecone.init).

# Usage (from the RAG folder):
#     python ingest_interactions.py --index energy-rag \
#         --model text-embedding-ada-002 \
#         --chunk-size 800 \
#         --overlap 120 \
#         --batch 50

# Dependencies:
#     pip install "pinecone-client>=3" langchain openai tiktoken
# """

# import sys
# from uuid import uuid4
# from datetime import datetime

# # ————————————————
# # 1) Hard-coded credentials (replace with your own keys)
# # ————————————————
# PINECONE_API_KEY = "pcsk_6esMas_Tfb3tkmUXKj51nQX7geiCV9s4oEyxe5q8w9EVx8EMpG9tWAkk3e98fYUbTv3RV3"
# PINECONE_ENV     = "us-east-1"  # e.g. "us-east-1"
# OPENAI_API_KEY   = "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"
# import os
# # Make sure LangChain/OpenAI picks up the key:
# os.environ["OPENAI_API_KEY"] = "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"


# import pinecone
# try:
#     # Import Pinecone v7 classes:
#     from pinecone import Pinecone, ServerlessSpec
# except ImportError:
#     sys.exit("Missing `pinecone-client>=3`. Install via `pip install pinecone-client`.")

# try:
#     from langchain.embeddings.openai import OpenAIEmbeddings
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain.schema import Document
# except ImportError:
#     sys.exit("Missing `langchain` (pip install langchain).")

# def initialize_pinecone_index(index_name: str, dimension: int):
#     """
#     Initialize a Pinecone v7 client and create the index if it doesn't exist.
#     Returns a handle to that index.
#     """
#     # Create a Pinecone client instance
#     pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

#     # List existing indexes (names())
#     existing = pc.list_indexes().names()
#     if index_name not in existing:
#         print(f"🆕  Creating index '{index_name}' (dim={dimension}, metric='cosine') …")
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#         )
#         # Wait until ready
#         while True:
#             desc = pc.describe_index(index_name)
#             if desc.status.ready:
#                 break

#     # Return an Index handle
#     return pc.Index(index_name)

# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Interactively ingest prompt/response pairs into a Pinecone index."
#     )
#     parser.add_argument(
#         "--index", required=True,
#         help="Name of the Pinecone index (e.g. energy-rag)"
#     )
#     parser.add_argument(
#         "--model", default="text-embedding-ada-002",
#         help="OpenAI embedding model (default: text-embedding-ada-002)"
#     )
#     parser.add_argument(
#         "--chunk-size", type=int, default=800,
#         help="Max characters per chunk (default: 800)"
#     )
#     parser.add_argument(
#         "--overlap", type=int, default=120,
#         help="Character overlap between chunks (default: 120)"
#     )
#     parser.add_argument(
#         "--batch", type=int, default=50,
#         help="Number of vectors per upsert batch (default: 50)"
#     )
#     args = parser.parse_args()

#     INDEX_NAME   = args.index
#     MODEL_NAME   = args.model
#     CHUNK_SIZE   = args.chunk_size
#     OVERLAP_SIZE = args.overlap
#     BATCH_SIZE   = args.batch

#     # ————————————————
#     # 2) Initialize embeddings & dimension (hard-code 1536 for ada-002)
#     # ————————————————
#     embedder = OpenAIEmbeddings(model=MODEL_NAME)
#     dimension = 1536  # text-embedding-ada-002 always returns 1536-dim vectors

#     # ————————————————
#     # 3) Initialize Pinecone index (using Pinecone v7 API)
#     # ————————————————
#     try:
#         index = initialize_pinecone_index(INDEX_NAME, dimension)
#     except Exception as e:
#         print(f"❌  Failed to initialize Pinecone index: {e}")
#         sys.exit(1)

#     # ————————————————
#     # 4) Prepare the text splitter
#     # ————————————————
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=OVERLAP_SIZE,
#     )

#     # ————————————————
#     # 5) Interactive ingestion loop
#     # ————————————————
#     print("\n🎯  Interactive Pinecone ingestion (hard-coded keys & Pinecone v7)")
#     print("   • To stop, leave the Prompt blank and press Enter.\n")

#     buffer = []
#     total_upserted = 0

#     while True:
#         prompt_text = input("Prompt (blank to quit) ► ").strip()
#         if prompt_text == "":
#             break

#         response_text = input("Response ► ").strip()
#         response_lines = []
#         while True:
#             line = input()
#             if line.strip() == "":
#                 break
#             response_lines.append(line)
#         response_text = "\n".join(response_lines).strip()

#         if response_text == "":
#             print("⚠️  You entered a Prompt but left Response blank. Skipping.\n")
#             continue

#         combined = f"User prompt:\n{prompt_text}\n\nAssistant response:\n{response_text}"
#         chunks = splitter.split_text(combined)
#         if not chunks:
#             chunks = [combined]

#         for chunk_id, chunk in enumerate(chunks):
#             vec = {
#                 "id": f"pair_{uuid4().hex[:8]}_{int(datetime.utcnow().timestamp())}_{chunk_id}",
#                 "values": embedder.embed_query(chunk),
#                 "metadata": {
#                     "prompt":    prompt_text,
#                     "response":  response_text,
#                     "timestamp": datetime.utcnow().isoformat(),
#                     "chunk_id":  chunk_id,
#                 },
#             }
#             buffer.append(vec)

#         if len(buffer) >= BATCH_SIZE:
#             index.upsert(buffer)
#             total_upserted += len(buffer)
#             print(f"✅  Upserted {total_upserted} vectors so far …\n")
#             buffer.clear()

#     if buffer:
#         index.upsert(buffer)
#         total_upserted += len(buffer)
#         print(f"✅  Upserted final batch. Total vectors upserted: {total_upserted}")

#     print(f"\n🏁  Done. Total prompt/response chunks upserted: {total_upserted}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
ingest_interactions.py

Interactively ask for prompt/response pairs and upsert them into a Pinecone index.
This version uses Pinecone v6.0.2 client (instantiating the Pinecone class directly).
"""

import sys
import os
import argparse
import time
from uuid import uuid4
from datetime import datetime

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

# ————————————————
# IMPORTANT: Use the v6.0.2 Pinecone client, not v7
# ————————————————
from pinecone import Pinecone
# (No pinecone.init(...) in v6.0.2)

from tiktoken import encoding_for_model

# ————————————————
# 1) Hard-coded credentials (replace with your own keys)
# ————————————————
PINECONE_API_KEY = "pcsk_6esMas_Tfb3tkmUXKj51nQX7geiCV9s4oEyxe5q8w9EVx8EMpG9tWAkk3e98fYUbTv3RV3"
PINECONE_ENV     = "us-east-1"  # e.g. "us-east-1" or whatever your environment is

# (Optional) override via environment variables:
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", PINECONE_API_KEY)
# PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT", PINECONE_ENV)


def initialize_pinecone_index(index_name: str, dimension: int):
    """
    Initialize a Pinecone v6.0.2 client and create the index if it doesn't exist.
    Returns a handle to that index.
    """
    # 1) Instantiate the Pinecone client (v6.x) with API key + environment
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # 2) Retrieve existing index names via the client
    existing = pc.list_indexes().names()

    if index_name not in existing:
        print(f"🆕  Creating index '{index_name}' (dim={dimension}, metric='cosine') …")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
            # (You can add spec=ServerlessSpec(cloud="aws", region="us-east-1") here if needed,
            #  but Pinecone v6.0.2 will default to the default spec if omitted.)
        )
        # Poll until the index appears in list_indexes()
        while True:
            if index_name in pc.list_indexes().names():
                break
            time.sleep(1)

    # 3) Connect to the index.  In v6.x, use pc.Index(index_name)
    idx = pc.Index(index_name)
    return idx


def get_encoding(model_name: str):
    """
    Get token encoding for a given model (e.g. "text-embedding-ada-002").
    """
    try:
        return encoding_for_model(model_name)
    except Exception:
        # Fallback if the specific model isn't found
        return encoding_for_model("cl100k_base")


def count_tokens(text: str, encoding_name):
    """
    Count the number of tokens in `text`, using the specified encoding.
    """
    return len(encoding_name.encode(text))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    """
    Split the input `text` into chunks of approximately `chunk_size` characters,
    with `chunk_overlap` characters of overlap between chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[""],  # only split by raw character limit
    )
    
    return splitter.split_text(text)


def embed_and_upsert(
    index,
    embedder,
    prompt: str,
    response: str,
    MODEL_NAME: str,
    token_limit: int,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
):
    """

    # Given a single prompt/response pair, split it into chunks, embed each chunk,
    # and upsert to the Pinecone index in batches of `batch_size`.
    # """
    encoding = get_encoding(MODEL_NAME)

    # Concatenate prompt and response into one text blob
    full_text = f"### Prompt:\n{prompt}\n\n### Response:\n{response}"

    # Split entire text into chunks by character length
    text_chunks = chunk_text(full_text, chunk_size, chunk_overlap)

    # Now embed each chunk and upsert
    vectors_to_upsert = []
    for chunk in text_chunks:
        # Generate a unique ID for each chunk
        chunk_id = str(uuid4())
        # Embed text chunk
        emb = embedder.embed_query(chunk)
        # Construct the metadata dictionary
        metadata = {
            "original_prompt": prompt,
            "original_response": response,
            "chunk_text": chunk,
            "timestamp": datetime.utcnow().isoformat(),
        }
        vectors_to_upsert.append((chunk_id, emb, metadata))

    # # Concatenate prompt and response into one text blob
    # full_text = f"### Prompt:\n{prompt}\n\n### Response:\n{response}"

    # # If this text is longer than token_limit, break into smaller chunks
    # n_tokens = count_tokens(full_text, encoding)
    # if n_tokens > token_limit:
    #     text_chunks = chunk_text(full_text, chunk_size, chunk_overlap)
    # else:
    #     text_chunks = [full_text]

    # # Now embed each chunk and upsert
    # vectors_to_upsert = []
    # for chunk in text_chunks:
    #     # Generate a unique ID for each chunk
    #     chunk_id = str(uuid4())
    #     # Embed text chunk
    #     emb = embedder.embed_query(chunk)
    #     # Construct the metadata dictionary
    #     metadata = {
    #         "original_prompt": prompt,
    #         "original_response": response,
    #         "chunk_text": chunk,
    #         "timestamp": datetime.utcnow().isoformat(),
    #     }
    #     vectors_to_upsert.append((chunk_id, emb, metadata))

        # When we have a full batch, upsert it
        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert.clear()

    # Upsert any remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively ingest prompt/response pairs into Pinecone index."
    )
    parser.add_argument(
        "--index", type=str, required=True, help="Name of the Pinecone index (e.g. energy-rag)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-ada-002",
        help="OpenAI embedding model name (default: text-embedding-ada-002)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target chunk size in characters (default: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Character overlap between chunks (default: 120)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Number of vectors per upsert batch (default: 50)",
    )
    args = parser.parse_args()

    INDEX_NAME = args.index
    MODEL_NAME = args.model
    CHUNK_SIZE = args.chunk_size
    OVERLAP_SIZE = args.overlap
    BATCH_SIZE = args.batch

    # 2) Initialize embeddings & dimension (1536 for ada-002)
    embedder = OpenAIEmbeddings(model=MODEL_NAME)
    dimension = 1536  # text-embedding-ada-002 → 1536‐dim vectors
    token_limit = 8191  # max tokens for ada-002

    # 3) Initialize Pinecone index (using Pinecone v6.0.2 API)
    try:
        index = initialize_pinecone_index(INDEX_NAME, dimension)
    except Exception as e:
        print(f"❌  Failed to initialize Pinecone index: {e}")
        sys.exit(1)

    # 4) Interactive loop: ask user for prompt/response pairs
    print("\n🎯  Interactive Pinecone ingestion (using Pinecone v6.0.2)")
    total_upserted = 0
    try:
        while True:
            print("\n--- New prompt/response pair ---")
            prompt = input("👉  Enter prompt (or type 'quit' to exit):\n")
            if prompt.strip().lower() == "quit":
                break
            print("👉  Enter response (finish with a blank line):")
            lines_resp = []
            while True:
                line = input()
                if line == "":
                    break
                lines_resp.append(line)
            response = "\n".join(lines_resp)


            print("\n🛠  Embedding & upserting…")
            embed_and_upsert(
                index=index,
                embedder=embedder,
                prompt=prompt,
                response=response,
                MODEL_NAME=MODEL_NAME,
                token_limit=token_limit,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=OVERLAP_SIZE,
                batch_size=BATCH_SIZE,
            )

            # Estimate how many chunks we just upserted
            n_chunks = len(chunk_text(f"### Prompt:\n{prompt}\n\n### Response:\n{response}", CHUNK_SIZE, OVERLAP_SIZE))
            total_upserted += n_chunks
            print(f"\n✅  Upserted {n_chunks} chunks (total so far: {total_upserted}).")

    except KeyboardInterrupt:
        print("\n🛑  Interrupted by user. Exiting…")

    print(f"\n🏁  Done. Total prompt/response chunks upserted: {total_upserted}")


if __name__ == "__main__":
    main()
