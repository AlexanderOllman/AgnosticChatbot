from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    openai_api_base="https://embeddings-v5-predictor-global-models.beta.hpepcai.com/v1",
    openai_api_key="1234",
)


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)