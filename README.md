# RAG Experiments

Implemented a RAG system in 3 different ways:

1. Using langchain/transorfmers library to run the RAG system - _locally_
2. Using Langroid library which uses OpenAI's API - running _remotely_ 
3. Using OpenAI's API directly, no other libraries used - running _remotely_

Im doing this to get a better understanding how the RAG system works and how to use it in different ways. But also what
are the benefits and limitations of the various libraries and APIs and what it takes to implement them.
    
In conclusion I think it's probably best to stick to OpenAIs API as much as possible and while it adds some extra work,
it avoids the extra layers of abstractions that langroid, but especially Langchain adds. Langroid was actually quite 
nice and I would probably prefer to work with it instead of Langchain given the choice.

### ENV variables you would need to get the magic working
Used for the local Langchain implementation
1. READER_MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
2. EMBEDDING_MODEL_NAME = "avsolatorio/GIST-Embedding-v0"
3. MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""]
4. HUG_TOKEN = YOUR_API_KEY


1. LOCAL_PDF_PATH = "path/to/pdf"
2. OPENAI_API_KEY=YOUR_API_KEY
3. PYTHONIOENCODING="utf-8"