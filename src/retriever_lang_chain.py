from langchain.docstore.document import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from transformers import AutoTokenizer
from typing import Optional, List
from dotenv import load_dotenv
from misc import load_pdf
import os

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
MARKDOWN_SEPARATORS = os.getenv('MARKDOWN_SEPARATORS')


def split_documents(chunk_size: int, knowledge_base: List[LangchainDocument],
                    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    #TODO: try diffrent txt splitters - https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS)

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique



def retriever_model(pdf_path, prompt):
    # load pdf and split into chunks
    pages = load_pdf(pdf_path)
    docs_processed = split_documents(512, pages, tokenizer_name=EMBEDDING_MODEL_NAME)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                            multi_process=False,
                                            model_kwargs={"trust_remote_code": True},
                                            encode_kwargs={"normalize_embeddings": True})

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(docs_processed, embedding_model,
                                                     distance_strategy=DistanceStrategy.COSINE)

    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=prompt, k=2)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # We only need the text of the documents

    #context = "\nExtracted documents:\n"
    context = "".join([f"Document {str(i)}:\n" + doc for i, doc in enumerate(retrieved_docs_text)])
    return context
