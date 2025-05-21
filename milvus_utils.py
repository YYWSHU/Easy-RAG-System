import streamlit as st
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from config import (
    COLLECTION_NAME, EMBEDDING_DIM,
    INDEX_TYPE, INDEX_PARAMS, INDEX_METRIC_TYPE,
    SEARCH_PARAMS, TOP_K, MAX_ARTICLES_TO_INDEX,
    id_to_doc_map
)
from sklearn.preprocessing import normalize  # 引入归一化模块


def init_milvus_connection(host: str = "localhost", port: str = "19530") -> str:
    try:
        connections.connect(alias="default", host=host, port=port)
        st.success("Connected to Milvus Standalone!")
        return "default"
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None


def get_or_create_collection(
        alias: str,
        collection_name: str,
        embedding_dim: int,
        index_type: str,
        index_params: dict,
        index_metric_type: str
) -> Collection:
    try:
        existing = utility.list_collections()
        if collection_name in existing:
            utility.drop_collection(collection_name)
            st.info(f"Dropped existing collection '{collection_name}' to apply updated schema.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields, description=f"RAG collection with dim={embedding_dim}")

        collection = Collection(name=collection_name, schema=schema, using=alias)
        st.success(f"Created collection '{collection_name}' with dim={embedding_dim}.")

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": "IP",  # ✅ 使用内积（Inner Product）
                "params": index_params
            }
        )
        st.info(f"Created index ({index_type}, metric_type=IP) on '{collection_name}'.")

        collection.load()
        st.success(f"Collection '{collection_name}' is loaded and ready.")
        return collection

    except Exception as e:
        st.error(f"Error setting up Milvus collection: {e}")
        return None


def index_data_if_needed(
        collection: Collection,
        data: list,
        embedding_model,
        max_articles: int = MAX_ARTICLES_TO_INDEX
) -> bool:
    if collection is None:
        st.error("Milvus collection not available.")
        return False

    try:
        current_count = collection.num_entities
    except Exception:
        current_count = 0
    st.write(f"Currently in collection: {current_count} entities.")

    to_index = data[:max_articles]
    docs, temp_map = [], {}

    for i, doc in enumerate(to_index):
        title = doc.get('title', '').strip()
        abstract = doc.get('abstract', '').strip()
        if not abstract:
            continue  # 忽略没有正文的条目

        content = f"Title: {title}\nAbstract: {abstract}" if title else f"Abstract: {abstract}"
        docs.append(content)
        temp_map[i] = {'title': title, 'abstract': abstract, 'content': content}

    needed = len(docs)
    if current_count < needed and docs:
        st.warning(f"Indexing {needed - current_count} new documents...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

        # 归一化处理
        embeddings = normalize(embeddings, norm='l2')  # L2 normalization

        ids = list(range(needed))
        previews = [c[:500] for c in docs]

        try:
            collection.insert([ids, embeddings, previews])
            collection.flush()
            id_to_doc_map.update(temp_map)
            st.success(f"Inserted {needed} documents into Milvus.")
            return True
        except Exception as e:
            st.error(f"Error inserting data: {e}")
            return False
    else:
        st.write("No new documents to index.")
        if not id_to_doc_map:
            id_to_doc_map.update(temp_map)
        return True


def search_similar_documents(
        collection: Collection,
        query: str,
        embedding_model
) -> tuple:
    if collection is None:
        st.error("Milvus collection not available.")
        return [], []

    try:
        q_emb = embedding_model.encode([query])[0]

        # 归一化处理
        q_emb = normalize([q_emb], norm='l2')[0]  # L2 normalization

        results = collection.search(
            data=[q_emb],
            anns_field="embedding",
            param={"metric_type": "IP", **SEARCH_PARAMS},  # ✅ 明确使用 IP
            limit=TOP_K,
            output_fields=["id"]
        )
        if not results or not results[0]:
            return [], []
        ids = [hit.entity.get("id") for hit in results[0]]
        dists = [hit.distance for hit in results[0]]
        return ids, dists
    except Exception as e:
        st.error(f"Search failed: {e}")
        return [], []
