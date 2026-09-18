'''
  Loads a PDF and splits it into chunks, using LangChain's document
  loader and text splitter instead of hand-written code.
'''

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentsPreparer:
  ''' Loads a PDF and splits it into chunks ready for embedding. '''

  def __init__(self, chunk_size=800, overlap=100):
    self.splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=overlap,
    )

  def pdf_loader(self, pdf_path):
    ''' Loads a PDF into a list of LangChain Document objects, one per page. '''
    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()

  def documents_splitter(self, documents):
    ''' Splits Document objects into smaller chunked Documents. '''
    return self.splitter.split_documents(documents)

  def pdf_preparer(self, pdf_path):
    ''' Convenience wrapper: load + split in one call. '''
    documents = self.pdf_loader(pdf_path)
    return self.documents_splitter(documents)