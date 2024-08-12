from pypdf import PdfReader
from typing import List, Optional
from scipy import spatial
import numpy as np

from typing import List, Optional
from scipy import spatial


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def load_pdf(file_path):
    pages = []
    reader = PdfReader(file_path)
    number_of_pages = len(reader.pages)

    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        pages.append(page.extract_text())
    return pages


def remove_newlines(text):
    text = text.replace('\n', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('•', '')
    text = text.lower()
    return text


def split_into_sentences(list_o_text):
    whole_text = ''
    for entry in list_o_text:
        cleaned_str = remove_newlines(entry)
        whole_text += cleaned_str
    sentences = whole_text.split('. ')
    return sentences
