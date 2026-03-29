import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# function for loading and chunking documents
def process_pdfs(uploaded_files):
    all_docs = []
    with tempfile.TemporaryDirectory() as tempdir:
        for uploaded_file in uploaded_files:
            temp_filepath = os.path.join(tempdir, uploaded_file.name)

            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyMuPDFLoader(temp_filepath)
            pages = loader.load() # each page is loaded separately 
            all_docs.extend(pages) 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunked_documents = text_splitter.split_documents(all_docs) # pages are chunked

    return chunked_documents
