#%%
import argparse
import pandas as pd
import os

from pymilvus import DataType, MilvusClient
from tqdm import tqdm

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

class EmbeddingFromDict(Embeddings):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        if isinstance(dictionary, str):
            print('Loading embeddings')
            self.dictionary = pd.read_pickle(dictionary)
        elif isinstance(dictionary, dict):
            self.dictionary = dictionary
        else:
            raise Exception("Cannot load embedding")
    
    def embed_documents(self, texts):
        res = [self.dictionary[text] for text in texts]
        return res
    
    def embed_query(self, text):
        return self.dictionary[text]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=str, required=True, help='specify input doc graph')
    parser.add_argument('--embs', type=str, required=True, help='specify embedding pkl')
    parser.add_argument('--colname', type=str, required=True, help='specify collection name')
    parser.add_argument('--batch_size', type=int, default=1000, help='specify upload batch size')
    args = parser.parse_args()
    
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "tcp://localhost:19530"))
    if client.has_collection(collection_name=args.colname):
        stats = client.get_collection_stats(collection_name=args.colname)
        if stats["row_count"] > 0:
            user_input = input("Collection exists with data. Recalculate embeddings? (yes/[no]): ")
            if user_input.lower() != "yes":
                return client
        client.drop_collection(collection_name=args.colname)
    del client
    

    print('Loading documents')
    G = pd.read_pickle(args.docs)

    vector_store = Milvus(
        embedding_function=EmbeddingFromDict(args.embs),
        collection_name=args.colname,
        connection_args={"uri": os.getenv("MILVUS_URI", "tcp://localhost:19530")},
        primary_field='id',
        text_field='doc',
        vector_field='embedding',
        index_params={
            "index_type": "FLAT",
            'metric_type': "L2",
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

    ids = [n for n in G.nodes()]
    texts = [d['doc_enc'] for _, d in G.nodes(data=True)]
    metadata = []
    for n, d in G.nodes(data=True):
        d.pop('clean_word_bag')
        d.pop('doc_enc')
        d['data_source'] = sorted(d['data_source'])
        metadata.append(d)
    for i in tqdm(range(0, len(G), args.batch_size), ncols=100, desc="Pushing"):
        vector_store.add_texts(texts[i:i+args.batch_size], metadata[i:i+args.batch_size], ids=ids[i:i+args.batch_size], batch_size=args.batch_size)

if __name__ == '__main__':
    main()

