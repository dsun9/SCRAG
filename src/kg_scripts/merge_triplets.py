#%%
import json
import pickle

import networkx as nx
from tqdm import tqdm


#%%
def load_graph():
    triplets = []
    with open("../../data/kg/chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            dic = json.loads(line)
            ts = dic["triplets"]

            for t in ts:
                if isinstance(t, dict) and len(t) == 3:
                    triplets.append(t)

    G = nx.DiGraph()
    for triplet in tqdm(triplets, ncols=100):
        if len(triplet) != 3:
            continue

        keys = list(triplet.keys())

        head = keys[0]
        relation = keys[1]
        tail = keys[2]

        head_type = triplet[head]
        tail_type = triplet[tail]

        head = head.replace("_", " ")
        tail = tail.replace("_", " ")
        relation = relation.replace("_", " ")
        if not G.has_node(head):
            G.add_node(head, kind=head_type)
        if not G.has_node(tail):
            G.add_node(tail, kind=tail_type)

        G.add_edge(head, tail, relation=relation)

    return G
#%%
if __name__ == "__main__":
    G = load_graph()
    print(G)
    with open("../../data/kg/kg.pkl", "wb") as f:
        pickle.dump(G, f)
