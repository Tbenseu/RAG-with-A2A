import os
from transformers import AutoModel, AutoTokenizer
os.environ["HF_HOME"] = "/tmp/huggingface"  # Custom cache path
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(["Your text here"]).tolist()

