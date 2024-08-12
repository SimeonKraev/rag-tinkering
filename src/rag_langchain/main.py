from src.reader import reader_model
from src.retriever_lang_chain import retriever_model
from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
LOCAL_PDF_PATH = os.getenv('LOCAL_PDF_PATH')

prompt = "What is anxiety?"

context = retriever_model(LOCAL_PDF_PATH, prompt)
print(f"Context len: {len(context)}")
answer = reader_model(context, prompt)

answer_formatted = textwrap.fill(answer, width=120)
print(f"Answer: {answer_formatted}")

# READER_MODEL_NAME = "TinyLlama/TinyLlama_v1.1"  # 1.1B params
# EMBEDDING_MODEL_NAME = "thenlper/gte-small" # "Snowflake/snowflake-arctic-embed-m" "avsolatorio/GIST-Embedding-v0"
# HUG_TOKEN = ""
# LOCAL_PDF_PATH = "C://Users//shush//Desktop//psych//02. Cognitive Therapy Skills Author University of Michigan.pdf"
# MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""]








