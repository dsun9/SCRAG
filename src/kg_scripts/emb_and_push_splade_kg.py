import torch
import pickle  # Use pickle instead of json
import os
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from pymilvus import MilvusClient, DataType
from pymilvus import model as milvus_model
from tqdm import tqdm
import collections
from random import shuffle
import re

# Configuration
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
DATA_PATH = "/data/shared/incas/neo4jdoc/coling25/data/knowledge_graph.pkl"
COLLECTION_NAME = "kg"

# Initialize Milvus client
def init_milvus():
    client = MilvusClient("kg3.db")
    
    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)


    
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    
    if client.has_collection(collection_name=COLLECTION_NAME):
        stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
        if stats["row_count"] > 0:
            user_input = input("Collection exists with data. Recalculate embeddings? (yes/[no]): ")
            if user_input.lower() != "yes":
                return client
        client.drop_collection(collection_name=COLLECTION_NAME)
    
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    return client

# Load SPLADE Model and Tokenizer
def load_splade_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Load Documents
def load_documents():
    documents = []
    with open(DATA_PATH, 'rb') as f:
        G = pickle.load(f)

    max_record = 0
    min_record = 1000000

    lens = []

    for node in tqdm(G.nodes(data=True), ncols=100):
        # if G.in_degree(node[0]) > 30 or G.out_degree(node[0]) > 30:
        #     continue
        triplet = []
        queue = collections.deque([node[0]])
        visited = set()

        while queue and len(triplet) < 15:
            for i in range(len(queue)):
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                # for successor in G.successors(n):
                #     triplet.append((n, G[n][successor]["relation"], successor))
                #     queue.append(successor)
                for predecessor in G.predecessors(n):
                    triplet.append((predecessor, G[predecessor][n]["relation"], n))
                    queue.append(predecessor)
                

        # Ordered dedup
        triplet = list(dict.fromkeys(triplet))
        if len(triplet) > 30:
            shuffle(triplet)
            triplet = triplet[:30]

        max_record = max(max_record, len(triplet))
        min_record = min(min_record, len(triplet))
        lens.append(len(triplet))
        
        documents.append("Sub knowledge graph:\n" + "\n".join([f"({t[0]}, {t[1]}, {t[2]})" for t in triplet]))
    print(documents[0])
    print(f"Max record: {max_record}")
    print(f"Min record: {min_record}")
    print(np.histogram(lens))

    
    
    return documents

# Insert documents into Milvus
def insert_into_milvus(client, documents, sparse_representations):
    print("🚀 Inserting document vectors into Milvus...")

    entities = [{
        "id": i, 
        "sparse": { coord: data for coord, data in zip(sparse_representations[i].coords[0], sparse_representations[i].data) }, 
        "content": documents[i].get("content", ""),  
    } for i in range(len(documents))]
    client.insert(collection_name=COLLECTION_NAME, data=entities)
    print("✅ Document representations ready in Milvus.") 

# Query Search Function
def search(client, query, tokenizer, model, top_k=2):
    query_rep = splade_ef.encode_queries([query])#
    # hf_query_rep = compute_splade_representation([query], tokenizer, model)

    # print("Query Representation: ", query_rep)
    # print("HF Query Representation: ", hf_query_rep)

    search_results = client.search(collection_name=COLLECTION_NAME, data=query_rep, limit=top_k, anns_field="sparse", output_fields=["id", "content"])
    return [(hit["entity"]["content"], hit["distance"]) for result in search_results for hit in result]

if __name__ == "__main__":
    
    with open("../../data/kg/kg.pkl", "rb") as f:
        G = pickle.load(f)
    
    
    client = init_milvus()
    tokenizer, model = load_splade_model()
    splade_ef = milvus_model.sparse.SpladeEmbeddingFunction(
        model_name="naver/splade-cocondenser-ensembledistil", 
        device="cuda:2"
    )
    # documents = load_documents()
    
    # if client.has_collection(collection_name=COLLECTION_NAME):
    #     stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    #     if stats["row_count"] == 0:
    #         print(f"Calculating SPLADE representations for {len(documents)} documents...")
            
    #         print("Calculating SPLADE representations done.")
    #         print("Inserting into Milvus...")
    #         for i in tqdm(range(0,len(documents),256)):
    #             docs = documents[i:i+256]
    #             sparse_representations = splade_ef.encode_documents(docs)#compute_splade_representation([doc["content"] for doc in documents], tokenizer, model)
    #             entities = [{
    #                 "sparse": {int(x): float(y) for x, y in sparse_representations[j].todok().items()} , 
    #                 "content": docs[j],
    #             } for j in range(len(docs))]
    #             client.insert(collection_name=COLLECTION_NAME, data=entities)
    
    # while True:
    #     query = input("Enter a query: ")
    #     if query.lower() == "exit":
    #         break
    #     results = search(client, query, tokenizer, model)
    #     for i, (doc, score) in enumerate(results, 1):
    #         print(f"Rank {i}: (Score: {score:.2f})")
    #         print("-" * 100)
    #         docarr = doc.split("\n")
    #         shuffle(docarr)
    #         docarr = docarr[:15]
    #         print(f"Content: {'\n'.join(docarr)}")
    #         print("=" * 100)

    data = json.load(open("/data/shared/incas/neo4jdoc/coling25/data/to_find_kg_reply.json"))
    kg_dic = {}

    def remove_at(text):
        pattern = r'@\w+(?:\s+@\w+)*'  # Matches @USER or @USER_LIST
        return re.sub(pattern, '', text).replace("@<USER_MENTION>", "").replace("@<USER_MENTION_LIST>", "").strip()

    for topic in data:
        for replies in data[topic]:
            id = replies[0]

            replies = replies[2]

            replies_str = "\n".join([remove_at(reply) for reply in replies])
            search_results = search(client, replies_str, tokenizer, model, top_k=1)
            kg_dic[id] = search_results[0][0]
    
    json.dump(kg_dic, open("/data/shared/incas/neo4jdoc/coling25/data/id2kg_map.json", "w"), indent=4)

            




        