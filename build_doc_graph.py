import json
import networkx as nx
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from config import EMBEDDING_MODEL_NAME, DATA_FILE

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    docs = json.load(f)

doc_texts = [doc.get("abstract", "") for doc in docs]
doc_ids = [doc.get("id") for doc in docs]
embeddings = embedding_model.encode(doc_texts, show_progress_bar=True, batch_size=32)
embeddings = normalize(embeddings)

# 构建图：每个文档为节点，相似度前 K 为边
G = nx.Graph()
top_k = 10
for i, emb in tqdm(enumerate(embeddings), total=len(embeddings)):
    G.add_node(doc_ids[i])
    sims = np.dot(embeddings, emb)
    top_idx = np.argsort(-sims)[1:top_k+1]
    for j in top_idx:
        G.add_edge(doc_ids[i], doc_ids[j], weight=float(sims[j]))

# 保存图
with open("doc_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
