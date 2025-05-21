import os
import time
import streamlit as st
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
import numpy as np

from config import (
    DATA_FILE,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    TOP_K,
    MAX_ARTICLES_TO_INDEX,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    INDEX_TYPE,
    INDEX_PARAMS,
    INDEX_METRIC_TYPE,
    SEARCH_PARAMS,
    id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from milvus_utils import (
    init_milvus_connection,
    get_or_create_collection,
    index_data_if_needed
)
from rag_core import generate_answer

# --- 环境 & UI 设置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME']     = './hf_cache'
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 with Iterative Feedback")
st.markdown(f"**Embedding 模型:** {EMBEDDING_MODEL_NAME}  •  **Generation 模型:** {GENERATION_MODEL_NAME}  •  **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- 1. 连接 Milvus & 准备 Collection ---
alias = init_milvus_connection(host="localhost", port="19530")
collection = get_or_create_collection(
    alias=alias,
    collection_name=COLLECTION_NAME,
    embedding_dim=EMBEDDING_DIM,
    index_type=INDEX_TYPE,
    index_params=INDEX_PARAMS,
    index_metric_type=INDEX_METRIC_TYPE
)

# --- 2. 加载模型 ---
embedding_model   = load_embedding_model(EMBEDDING_MODEL_NAME)
generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = load_reranker()

# --- 3. 索引数据（如有必要） ---
data = load_data(DATA_FILE)
index_data_if_needed(
    collection=collection,
    data=data,
    embedding_model=embedding_model,
    max_articles=MAX_ARTICLES_TO_INDEX
)

st.divider()

# --- 4. Graph-Augmented 检索 + Reranking ---
def search_with_reranking(query: str,
                      top_n: int = 50,
                      hops: int = 1,
                      per_hop: int = 10,
                      final_k: int = TOP_K):
    # 第一阶段：Milvus 向量召回 top_n
    st.info("阶段1：向量召回中…")
    q_emb = embedding_model.encode([query])[0]
    q_emb = normalize([q_emb], norm='l2')[0]
    res1 = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "IP", **SEARCH_PARAMS},
        limit=top_n,
        output_fields=["id"]
    )
    if not res1 or not res1[0]:
        return [], [], []
    cand = { hit.entity.get("id") for hit in res1[0] }
    milvus_scores = { hit.entity.get("id"): hit.distance for hit in res1[0] }

    # 第二阶段：文档–文档“图”扩展（hops 跳）
    st.info(f"阶段2：图扩展 {hops} 跳，每跳 top {per_hop}…")
    frontier = list(cand)
    for _ in range(hops):
        new_front = []
        for did in frontier:
            # 用文档本身 embedding 去 Milvus 搜 neighbors
            doc_emb = embedding_model.encode([ id_to_doc_map[did]["content"] ])[0]
            doc_emb = normalize([doc_emb], norm='l2')[0]
            neigh = collection.search(
                data=[doc_emb],
                anns_field="embedding",
                param={"metric_type": "IP", **SEARCH_PARAMS},
                limit=per_hop,
                output_fields=["id"]
            )
            for hit in neigh[0]:
                nid = hit.entity.get("id")
                if nid not in cand:
                    cand.add(nid)
                    milvus_scores[nid] = hit.distance
                    new_front.append(nid)
        frontier = new_front

    cand_ids = list(cand)
    # 准备 Cross-Encoder 二阶段排序对
    st.info("阶段3：Cross-Encoder Rerank…")
    pairs = [(query, id_to_doc_map[_id]["content"]) for _id in cand_ids]
    rerank_scores = reranker.predict(pairs, convert_to_numpy=True)

    # 取 final_k
    idxs = np.argsort(-rerank_scores)[:final_k]
    final_ids    = [cand_ids[i]     for i in idxs]
    final_scores = [float(rerank_scores[i]) for i in idxs]
    final_milvus = [milvus_scores[final_ids[i]] for i in range(len(final_ids))]

    return final_ids, final_milvus, final_scores

# --- 5. 会话状态 ---
if "round" not in st.session_state:
    st.session_state.round = 1
    st.session_state.history = []
    st.session_state.done = False
    st.session_state.await_feedback = False

def run_one_round(query):
    st.session_state.history.append(query)
    full_query = "；".join(st.session_state.history)  # 或 "\n".join(...) 更自然
    ids, dists, reranks = search_with_reranking(full_query)

    if not ids:
        st.warning("未检索到相关文档。")
        return

    st.success(f"第 {st.session_state.round} 轮检索完成，展示结果如下：")
    for i, (doc_id, d, r) in enumerate(zip(ids, dists, reranks), start=1):
        doc = id_to_doc_map.get(doc_id, {})
        title = doc.get("title", "无标题")
        with st.expander(f"{i}. {title}  •  Milvus距: {d:.4f}  •  Rerank得分: {r:.4f}"):
            st.write(doc.get("abstract", ""))
    st.info("正在生成答案…")
    contexts = [id_to_doc_map[i] for i in ids]
    answer = generate_answer(full_query, contexts, generation_model, tokenizer)
    st.subheader("📝 回答")
    st.write(answer)

# --- 6. 多轮交互 UI ---
if not st.session_state.done:
    label = f"第 {st.session_state.round} 轮，请输入您的问题"
    if st.session_state.round > 1:
        st.markdown(f"历史输入：{' ➔ '.join(st.session_state.history)}")
    user_input = st.text_input(label, key=f"q{st.session_state.round}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("提交检索", key=f"submit{st.session_state.round}"):
            if user_input.strip():
                run_one_round(user_input.strip())
                if st.session_state.round < 3:
                    st.session_state.await_feedback = True
                else:
                    st.success("已达到三轮上限，结束会话。")
                    st.session_state.done = True

    if st.session_state.await_feedback:
        with col2:
            if st.button("满意，结束", key=f"sat{st.session_state.round}"):
                st.success("感谢您的反馈，会话结束。")
                st.session_state.done = True
                st.session_state.await_feedback = False
            if st.button("不满意，继续", key=f"unsat{st.session_state.round}"):
                st.session_state.round += 1
                st.session_state.await_feedback = False
                st.rerun()

# --- 7. 侧边栏配置信息 ---
st.sidebar.header("系统配置")
st.sidebar.markdown(f"- 向量存储：Milvus Standalone")
st.sidebar.markdown(f"- Collection：{COLLECTION_NAME}")
st.sidebar.markdown(f"- 数据文件：{DATA_FILE}")
st.sidebar.markdown(f"- 最大索引数：{MAX_ARTICLES_TO_INDEX}")
st.sidebar.markdown(f"- 检索 Top K：{TOP_K}")
st.sidebar.markdown(f"- 一阶段召回 top_n：50")
st.sidebar.markdown(f"- 二阶段 Reranker：cross-encoder/ms-marco-MiniLM-L-6-v2")