import json
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pygraphviz as pgv

def load_graph(path) -> nx.nx_agraph:
    graph = pgv.AGraph(path)
    G_nx = nx.nx_agraph.from_agraph(graph)
    return G_nx

def vectorize_graph(graph):
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_vectors = model.wv
    return node_vectors

def traverse_and_extract_features(node, features):
    node_type = node.get('name', 'Unknown')
    # Increment the count of this node type in the features dictionary
    features[node_type] = features.get(node_type, 0) + 1
    
    # Recursively process children
    for child in node.get('children', []):
        traverse_and_extract_features(child, features)



def graph_embedding(graph, dimension=64):
    # embedding graph to 1xdimension vector
    node2vec = Node2Vec(graph, dimensions=dimension, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = [model.wv[str(node)] for node in graph.nodes()]
    embeddings_array = np.array(node_embeddings)
    graph_vector = np.mean(embeddings_array, axis=0)
    return graph_vector
    
def extract_ast_feature(json_path):
    ast = json.load(open(json_path))
    features = {}
    traverse_and_extract_features(ast, features)
    return features