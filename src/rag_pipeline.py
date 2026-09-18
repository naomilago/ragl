'''
  Top-level orchestrator: ingest a PDF, then answer questions grounded
  in it, using DocumentsPreparer + VectorStoreManager + ChatOllama.
'''

from langchain_ollama import ChatOllama

from src import config
from src.documents_preparer import DocumentsPreparer
from src.vector_store_manager import VectorStoreManager


SYSTEM_PROMPT = (
  'You are a helpful assistant that answers questions using ONLY the '
  "provided context. If the answer isn't in the context, say you don't know "
  'instead of guessing.'
)


class RagPipeline:
  ''' Ties together document prep, storage and generation. '''

  def __init__(self):
    self.preparer = DocumentsPreparer(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    self.store = VectorStoreManager(config.EMBED_MODEL, config.CHROMA_DIR, config.COLLECTION_NAME)
    self.llm = ChatOllama(model=config.LLM_MODEL, reasoning=False)

  def pdf_ingester(self, pdf_path, reset=True):
    ''' Loads, splits, embeds and stores a PDF. Returns the chunk count. '''
    if reset:
      self.store.collection_resetter()
    chunks = self.preparer.pdf_preparer(pdf_path)
    if not chunks:
      raise ValueError('No extractable text found in this PDF.')
    self.store.documents_indexer(chunks)
    return len(chunks)

  def question_asker(self, question, top_k=None):
    ''' Retrieves context and asks the LLM to answer using only that context. '''
    top_k = top_k or config.TOP_K
    retrieved = self.store.chunks_retriever(question, top_k)

    context = '\n\n---\n\n'.join(doc.page_content for doc in retrieved)
    pages = sorted({doc.metadata['page'] for doc in retrieved})

    messages = [
      ('system', SYSTEM_PROMPT),
      ('user', f'Context:\n{context}\n\nQuestion: {question}'),
    ]
    response = self.llm.invoke(messages)

    return {
      'answer': response.content,
      'pages': pages,
      'chunks': [doc.page_content for doc in retrieved],
    }