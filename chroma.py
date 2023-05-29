import environment

import json
import logging
import os
import re

import chromadb
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, ".chroma")


def replace_newlines_and_spaces(text):
    # Replace all newline characters with spaces
    text = text.replace("\n", " ")

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text


def get_documents():
    return PyPDFLoader("/Users/loctv/Documents/JHEP09(2017)103.pdf").load()


def init_chromadb():
    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embedding,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    documents = []
    for num, doc in enumerate(get_documents()):
        doc.page_content = replace_newlines_and_spaces(doc.page_content)
        documents.append(doc)

    vectorstore.add_documents(documents=documents, embedding=embedding)
    vectorstore.persist()
    print(vectorstore)


def query_chromadb():
    if not os.path.exists(DB_DIR):
        raise Exception(f"{DB_DIR} does not exist, nothing can be queried")

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embedding,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    result = vectorstore.similarity_search_with_score(query="who is?", k=4)
    jsonable_result = jsonable_encoder(result)
    print("-----------------------\n{0}\n-----------------------".format(json.dumps(jsonable_result, indent=2)))


def main():
    init_chromadb()
    query_chromadb()


if __name__ == '__main__':
    main()