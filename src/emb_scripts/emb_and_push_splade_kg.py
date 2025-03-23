import argparse
import pickle
import os
from math import ceil
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
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

def block_lists(list1, list2, N):
    result = []
    i, j = 0, 0
    len1, len2 = len(list1), len(list2)

    while i < len1 or j < len2:
        remaining1, remaining2 = len1 - i, len2 - j

        block = []

        if remaining1 > 0 and remaining2 > 0:
            half_N = N // 2
            take1 = min(half_N, remaining1)
            take2 = min(N - take1, remaining2)

            # Rebalance if either list can't fulfill half quota
            if take1 + take2 < N:
                extra_needed = N - (take1 + take2)
                if remaining1 - take1 > remaining2 - take2:
                    extra_from_1 = min(extra_needed, remaining1 - take1)
                    take1 += extra_from_1
                    extra_needed -= extra_from_1
                    take2 += min(extra_needed, remaining2 - take2)
                else:
                    extra_from_2 = min(extra_needed, remaining2 - take2)
                    take2 += extra_from_2
                    extra_needed -= extra_from_2
                    take1 += min(extra_needed, remaining1 - take1)

            block.extend(list1[i:i+take1])
            block.extend(list2[j:j+take2])
            i += take1
            j += take2

        elif remaining1 > 0:
            take1 = min(N, remaining1)
            block.extend(list1[i:i+take1])
            i += take1

        elif remaining2 > 0:
            take2 = min(N, remaining2)
            block.extend(list2[j:j+take2])
            j += take2

        result.append(block)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='specify batch size')
    parser.add_argument('--input', type=str, required=True, help='specify input graph')
    parser.add_argument('--colname', type=str, default="news_kg", help='specify collection name')
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
    
    with open(args.input, "rb") as f:
        G = pickle.load(f)
    chunks = []
    for n in G.nodes():
        node_docs_a = []
        node_docs_b = []
        out_deg = G.out_degree(n)
        in_deg = G.in_degree(n)
        if out_deg >= 10 or in_deg >= 10:
            for succ in G.successors(n):
                node_docs_a.append(f"({n})-[{G.edges[(n, succ)]['relation']}]->({succ})")
            for pred in G.predecessors(n):
                node_docs_b.append(f"({pred})-[{G.edges[(pred, n)]['relation']}]->({n})")
            if (out_deg + in_deg) <= 15:
                step_size = 15
            else:
                step_size = ceil((out_deg + in_deg) / ceil((out_deg + in_deg) / 10))
            chunks.extend(block_lists(node_docs_a, node_docs_b, step_size))
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
        }
    )
    for i in tqdm(range(0, len(chunks), args.batch_size), ncols=100, desc="Encoding"):
        texts = ['\n'.join(doc) for doc in chunks[i:i+args.batch_size]]
        vector_store.add_texts(texts, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
