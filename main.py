from src.reader import reader_model
from src.retriever import retriever_model
from dotenv import load_dotenv
import os

load_dotenv()
LOCAL_PDF_PATH = os.getenv('LOCAL_PDF_PATH')

prompt = "Give me a list of coping strategies for anxiety?"

context = retriever_model(LOCAL_PDF_PATH, prompt)
print(f"Context len: {len(context)}")
answer = reader_model(context, prompt)

print(f"Answer: {answer}")







