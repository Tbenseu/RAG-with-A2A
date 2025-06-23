import os
import hashlib
import json
import numpy as np
import logging
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

from transformers import AutoModel, AutoTokenizer


class DocumentTool:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.all_chunks = []
        self.chunk_to_doc = {}
        self.cache_file = 'semantic_cache.json'
        self.cache = self.load_cache()
        self.pdf_directory = './data/input_files/'
        os.makedirs(self.pdf_directory, exist_ok=True)
        
        self.chroma_client = chromadb.HttpClient(
        host="localhost",
        port=5000,
        settings=Settings()
        )
        
        # Get or create the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )

    def handle_uploaded_file(self, uploaded_file, filename=None):
        """
        Save uploaded file to disk and trigger processing.
        `uploaded_file` should be a file-like object (e.g., from Streamlit).
        """
        if not filename:
            filename = uploaded_file.name
        save_path = os.path.join(self.pdf_directory, filename)

        # Save uploaded file
        with open(save_path, 'wb') as out_file:
            out_file.write(uploaded_file.read())

        logging.info(f"File saved to: {save_path}")
        # Reprocess the PDFs now that there's a new file
        self.process_pdfs()

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text

    def create_chunks(self, text, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_files_hash(self):
        hash_md5 = hashlib.md5()
        for filename in sorted(os.listdir(self.pdf_directory)):
            if filename.endswith('.pdf'):
                with open(os.path.join(self.pdf_directory, filename), "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def process_pdfs(self):
        current_hash = self.get_files_hash()
        if self.cache.get('files_hash') == current_hash and self.collection.count() > 0:
            # No need to load index in ChromaDB as it's persistent
            return

        all_chunks = []
        chunk_to_doc = {}
        documents = []
        metadatas = []
        ids = []
        
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)
                text = self.extract_text_from_pdf(pdf_path)
                chunks = self.create_chunks(text)
                all_chunks.extend(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_{i}"
                    chunk_to_doc[chunk] = filename
                    documents.append(chunk)
                    metadatas.append({"source": filename})
                    ids.append(chunk_id)
        
        if documents:
            embeddings = self.model.encode(documents).tolist()
            
            # Clear existing collection
            self.chroma_client.delete_collection("document_chunks")
            self.collection = self.chroma_client.get_or_create_collection(
                name="document_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add documents to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        self.all_chunks = all_chunks
        self.chunk_to_doc = chunk_to_doc
        self.cache['files_hash'] = current_hash
        self.save_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                if cache.get('model_name') != self.model_name:
                    logging.info("Embedding model changed. Resetting cache.")
                    return {"queries": [], "embeddings": [], "responses": [], "model_name": self.model_name}
                return cache
        except FileNotFoundError:
            return {"queries": [], "embeddings": [], "responses": [], "model_name": self.model_name}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def retrieve_from_cache(self, query_embedding, threshold=0.5):
        for i, cached_embedding in enumerate(self.cache['embeddings']):
            if len(cached_embedding) != len(query_embedding):
                logging.warning("Cached embedding dimension mismatch. Skipping cache entry.")
                continue
            distance = np.linalg.norm(query_embedding - np.array(cached_embedding))
            if distance < threshold:
                return self.cache['responses'][i]
        return None

    def update_cache(self, query, query_embedding, response):
        self.cache['queries'].append(query)
        self.cache['embeddings'].append(query_embedding.tolist())
        self.cache['responses'].append(response)
        self.cache['model_name'] = self.model_name
        self.save_cache()

    def retrieve_relevant_chunks(self, query, top_k=10):
        query_vector = self.model.encode([query])[0]

        cached_response = self.retrieve_from_cache(query_vector)
        if cached_response:
            logging.info("Answer recovered from Cache.")
            return cached_response

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )
        
        relevant_chunks = results['documents'][0] if results['documents'] else []

        self.update_cache(query, query_vector, relevant_chunks)
        return relevant_chunks
    


# to run chromadb, use ```chroma run --path ./data --port 5000```
