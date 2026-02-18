import networkx as nx
import numpy as np
import random

def calculate_spectral_radius(G):
    if len(G.edges()) == 0: return 0
    adj = nx.to_numpy_array(G)
    return max(np.linalg.eigvals(adj).real)

G_original = nx.read_graphml("data/knowledge_graph.graphml")
initial_radius = calculate_spectral_radius(G_original)
print(f"Initial Radius: {initial_radius:.4f}")
