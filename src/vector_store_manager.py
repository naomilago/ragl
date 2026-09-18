'''
  Manages the Chroma vector store: embedding documents with Ollama and
  retrieving the closest chunks for a question.
'''

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
  ''' Wraps Ollama embeddings + a persistent Chroma collection. '''

  def __init__(self, embed_model, persist_directory, collection_name):
    self.persist_directory = str(persist_directory)
    self.collection_name = collection_name
    self.embeddings = OllamaEmbeddings(model=embed_model)
    self.store = self._store_builder()

  def _store_builder(self):
    return Chroma(
      collection_name=self.collection_name,
      embedding_function=self.embeddings,
      persist_directory=self.persist_directory,
    )

  def documents_indexer(self, documents):
    ''' Embeds and stores a list of chunked Documents. '''
    self.store.add_documents(documents)

  def chunks_retriever(self, query, top_k=4):
    ''' Returns the top_k Documents most similar to the query. '''
    return self.store.similarity_search(query, k=top_k)

  def collection_resetter(self):
    ''' Wipes the collection. Call before ingesting a new PDF. '''
    self.store.delete_collection()
    self.store = self._store_builder()