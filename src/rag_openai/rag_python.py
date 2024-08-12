import pandas as pd
import tiktoken
from misc import load_pdf, split_into_sentences, distances_from_embeddings
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np
from ast import literal_eval


load_dotenv()
LOCAL_PDF_PATH = os.getenv('LOCAL_PDF_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

pages = load_pdf(LOCAL_PDF_PATH)
# print(pages)

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
max_tokens = 500


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


sentences = split_into_sentences(pages)
df = pd.DataFrame(sentences, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

shortened = []
# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])

df = pd.DataFrame(shortened, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df['embeddings'] = df.text.apply(
        lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)
# df.to_csv('embeddings.csv')


def create_context(question, df, max_len=1800):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding
    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(df, question, max_len=1800, debug=False):
    context = create_context(question, df, max_len=max_len)
    msg = [{"role": "system", "content": "You are a helpful assistant."},
           {"role": "user",
            "content": f"Answer the question based on the context below: Context: {context}\n\n---\n\nQuestion: {
                       question}\nAnswer:"}]

    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = client.chat.completions.create(model="gpt-4o-mini", messages=msg)
        return response.choices[0].message.content

    except Exception as e:
        print(e)
        return ""


df = pd.read_csv('embeddings.csv', usecols=['text', 'n_tokens', 'embeddings'])
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
question = "How do we best treat anxiety?"
answer = answer_question(df, question, debug=True)
print(f"ANSWER: {answer}")


