from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from src.misc.misc import load_pdf
from dotenv import load_dotenv
import langroid as lr
import textwrap
import os

load_dotenv()
MARKDOWN_SEPARATORS = os.getenv('MARKDOWN_SEPARATORS')
PYTHONIOENCODING = os.getenv("PYTHONIOENCODING")

# system message
msg = ("Using the information contained in the extracts, give a comprehensive answer to the question.")
      # "Respond only to the question asked, response should be clear, unrepetative and relevant to the question.")
# PROMPT
prompt = "What causes anxiety and how can we treat it?"


# Load PDF
pages = load_pdf(os.getenv("LOCAL_PDF_PATH"))
# convert to Document objects
pages = [lr.Document(content=f"""{page}""", metadata=lr.DocMetaData(source="random-stuff-ftw")) for page in pages]

oai_embed_config = OpenAIEmbeddingsConfig(model_type="openai", model_name="text-embedding-ada-002", dims=1536)

llm_cfg = OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4_TURBO)

config = DocChatAgentConfig(llm=lr.language_models.OpenAIGPTConfig(chat_model=lr.language_models.OpenAIChatModel.GPT4),
                            vecdb=lr.vector_store.QdrantDBConfig(collection_name="quick-start-chat-agent-docs",
                                                                 replace_collection=True),
                            parsing=lr.parsing.parser.ParsingConfig(separators=["\n\n"],
                                                                    splitter=lr.parsing.parser.Splitter.SIMPLE,
                                                                    n_similar_docs=3),
                            system_message=msg)
agent = DocChatAgent(config)
agent.ingest_docs(pages)

response = agent.llm_response(prompt)

answer_formatted = textwrap.fill(response.__str__(), width=120)
print(f"Answer: {answer_formatted}")