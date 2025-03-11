import argparse
import os
import pickle

from neo4j import GraphDatabase
from tqdm import tqdm

# Neo4j connection details
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def load_graph(path):
    """Load a NetworkX graph from a pickle file."""
    with open(path, 'rb') as f:
        G = pickle.load(f)
    print("Graph loaded successfully!")
    print(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")
    return G

def upload_to_neo4j(G):
    """Connect to Neo4j and upload the graph."""
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        # session.execute_write(insert_graph, graph)
        tx = session.begin_transaction()
        count = 0
        for node, attrs in tqdm(G.nodes(data=True), desc="Inserting Nodes"):
            attrs.pop('clean_word_bag')
            attrs.pop('doc_enc')
            attrs.pop('kind')
            attrs['data_source'] = list(attrs['data_source'])
            tx.run(
                "MERGE (n:DOC {id: $id}) SET n += $properties",
                id=node, properties=attrs
            )
            count += 1
            if count % 1000 == 0:
                tx.commit()  # Explicitly commit the transaction (flush)
                print(f"Committed {count} nodes")
                tx = session.begin_transaction()  # Start a new transaction
        tx.commit()
        print(f"Final commit after inserting {count} nodes")
        
        count = 0
        tx = session.begin_transaction()
        print(f"Final commit after inserting {count} nodes")
        for u, v, attrs in tqdm(G.edges(data=True), desc="Inserting Edges"):
            tx.run(
                """
                MATCH (a:DOC {id: $id1}), (b:DOC {id: $id2})
                MERGE (a)-[r:RESPONSE]->(b)
                SET r += $properties
                """,
                id1=u, id2=v, properties=attrs
            )
            count += 1
            if count % 1000 == 0:
                tx.commit()  # Explicit flush for edges
                print(f"Committed {count} edges")
                tx = session.begin_transaction()
        tx.commit()
        print(f"Final commit after inserting {count} edges")
    print("Graph successfully inserted into Neo4j.")
    driver.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Specify input NetworkX pickle file')
    args = parser.parse_args()

    G = load_graph(args.input)
    upload_to_neo4j(G)

if __name__ == '__main__':
    main()
