import asyncio
from typing import List
from langchain import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.base import BaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import re
import sys,os
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from sys import argv
from langchain.vectorstores import Pinecone
import pinecone
import streamlit as st

cwd = os.getcwd()
st.secrets.load_if_toml_exists()

PERSIST_DIRECTORY="OA"
EMBEDDING = OpenAIEmbeddings()
PINECONE_INDEX_NAME = "axelerant-oa"

os.environ['OPENAI_KEY'] = st.secrets['OPENAI_API_KEY']


pinecone.init(
    api_key=st.secrets['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=st.secrets['PINECONE_ENV']  # next to api key in console
)

## Not in use, can be used to make a local index
def doc_loader_chroma():
    urls = open("data/axelerant.txt").read().splitlines()
    urls = [f"./data/axelerant/" + s for s in urls]
    loaders = []
    for file in urls:
        print (file)
        loaders.append(UnstructuredHTMLLoader(file))


    class MyVectorstoreIndexCreator(VectorstoreIndexCreator):
        def __init(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def from_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
            """Create a vectorstore index from loaders."""
            docs = []
            for loader in loaders:
                try:
                    docs.extend(loader.load())
                except:
                    print("failed")
                    print(loader)
            sub_docs = self.text_splitter.split_documents(docs)
            vectorstore = self.vectorstore_cls.from_documents(
                sub_docs, self.embedding, **self.vectorstore_kwargs
            )
            return VectorStoreIndexWrapper(vectorstore=vectorstore)
    vic = MyVectorstoreIndexCreator(vectorstore_cls = Chroma, vectorstore_kwargs={"persist_directory": PERSIST_DIRECTORY})
    vic.from_loaders(loaders)

def doc_loader_pinecone():
    urls = open("data/axelerant.txt").read().splitlines()
    #urls = [f"file://{cwd}/data/axelerant/" + s for s in urls]
    urls = [f"./data/axelerant/" + s for s in urls]
    #urls = [re.split("(.*)\?",s)[1] for s in urls]

    loaders = []
    #urls = urls[4:7]
    for file in urls:
        if (os.path.getsize(file) == 0):
            print(f"Skipping {file} because it is empty")
            continue;
        loaders.append(UnstructuredHTMLLoader(file))


    class MyVectorstoreIndexCreator(VectorstoreIndexCreator):
        def __init(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def from_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
            """Create a vectorstore index from loaders."""
            docs = []
            for loader in loaders:
                try:
                    docs.extend(loader.load())
                except:
                    print("failed")
                    print(loader)
            sub_docs = self.text_splitter.split_documents(docs)
            vectorstore = self.vectorstore_cls.from_documents(
                sub_docs, self.embedding, **self.vectorstore_kwargs
            )
            return VectorStoreIndexWrapper(vectorstore=vectorstore)
    vic = MyVectorstoreIndexCreator(vectorstore_cls = Pinecone, vectorstore_kwargs={"index_name": PINECONE_INDEX_NAME})
    vic.from_loaders(loaders)

async def delete_index():
    print("deleting index")
    return pinecone.Index(PINECONE_INDEX_NAME).delete(delete_all=True)


async def main():
    # Commented out
    await delete_index()
    doc_loader_pinecone()

asyncio.run(main())