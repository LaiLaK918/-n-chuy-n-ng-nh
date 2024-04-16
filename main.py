import networkx as nx
import matplotlib.pyplot as plt
from slither import Slither
from utils import load_graph, graph_embedding, extract_ast_feature
import json
import os
import pandas as pd

def generate_ast_data():
    ast_folder = "ast"
    df_data = []
    for root, dirs, files in os.walk(ast_folder):
        for file in files:
            file_path = os.path.join(root, file)
            res = extract_ast_feature(file_path)
            contract = file.split('.')[0]
            res['contract'] = contract
            df_data.append(res)
    df = pd.DataFrame(df_data)
    print(df)
    df.to_csv("csv/ast.csv", index=False)
    
def generate_cg_data():
    import glob
    cg_path = "contracts/*all_contracts.call-graph.dot"
    df_data = []
    for file_path in glob.glob(cg_path):        
        graph = load_graph(file_path)
        vector = graph_embedding(graph)
        dict_vector = {i: val for i, val in enumerate(vector)}
        contract = file_path.split('.')[0].split('/')[-1]
        dict_vector['contract'] = contract
        df_data.append(dict_vector)
    df = pd.DataFrame(df_data)
    print(df)
    df.to_csv("csv/cg.csv", index=False)
if __name__ == '__main__':
    generate_ast_data()
    generate_cg_data()
