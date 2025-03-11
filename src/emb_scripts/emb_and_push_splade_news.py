import argparse
import json
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType, MilvusClient
from pymilvus import model as milvus_model
from tqdm import tqdm


class SpladeEmbedding(Embeddings):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.splade_ef = milvus_model.sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-ensembledistil", 
            device=device
        )
    def embed_documents(self, texts):
        outputs = self.splade_ef.encode_documents(texts)
        ret = [{int(x): float(y) for x, y in output.todok().items()} for output in outputs]
        return ret
    def embed_query(self, text):
        return self.embed_documents([text])[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='specify batch size')
    parser.add_argument('--input', type=str, required=True, help='specify input jsonl')
    parser.add_argument('--colname', type=str, default="news_splade", help='specify collection name')
    parser.add_argument('--device', type=str, default='cpu', help='specify device')
    args = parser.parse_args()
    args.input = Path(args.input).resolve()
    
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "tcp://localhost:19530"))
    if client.has_collection(collection_name=args.colname):
        stats = client.get_collection_stats(collection_name=args.colname)
        if stats["row_count"] > 0:
            user_input = input("Collection exists with data. Recalculate embeddings? (yes/[no]): ")
            if user_input.lower() != "yes":
                return client
        client.drop_collection(collection_name=args.colname)
    del client
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
    chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for l in f:
            tmp = json.loads(l)
            splits = text_splitter.split_text(tmp['content'])
            for j, doc in enumerate(splits):
                new_record = dict(**tmp)
                new_record["content"] = doc
                new_record["chunk_seq"] = j
                new_record["total_chunk"] = len(splits)
                chunks.append(new_record)
    print("Loaded", len(chunks), "documents")
    
    vector_store = Milvus(
        embedding_function=SpladeEmbedding(args.device),
        collection_name=args.colname,
        connection_args={"uri": os.getenv("MILVUS_URI", "tcp://localhost:19530")},
        auto_id=True,
        primary_field='id',
        text_field='doc',
        vector_field='embedding',
        index_params={
            "field_name": "embedding",
            "index_type": "SPARSE_INVERTED_INDEX",
            'metric_type': "IP",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        },
        search_params={
            'metric_type': "IP",
            'params': {},
        },
        vector_schema={
            "dtype": DataType.SPARSE_FLOAT_VECTOR
        },
        metadata_schema={
            "data_source": {
                'dtype': DataType.ARRAY, 
                'element_type': DataType.VARCHAR, 
                'max_capacity': 8, 
                'max_length': 32,
            }
        }
    )
    for i in tqdm(range(0, len(chunks), args.batch_size), ncols=100, desc="Encoding"):
        texts = [doc['content'] for doc in chunks[i:i+args.batch_size]]
        metadata = [{
            "url": doc["url"],
            "title": doc.get("title", None) or "",
            "author": doc.get("author", None) or "",
            "published": doc["published"],
            "data_source": doc["data_source"],
            "total_chunk": doc["total_chunk"],
            "chunk_seq": doc["chunk_seq"],
        } for doc in chunks[i:i+args.batch_size]]
        vector_store.add_texts(texts, metadata, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
