import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Load the verified 731-node graph
G = nx.read_graphml("data/knowledge_graph.graphml")

# Calculate 3D Layout
pos = nx.spring_layout(G, dim=3, seed=42)

# Extract node coordinates and attributes
node_x, node_y, node_z = [], [], []
node_text, node_size = [], []

# Highlight specific hubs identified in the report
highlight_nodes = {"Liminality": "red", "Dissolution": "blue", "Site Percolation": "green"}

for node in G.nodes():
    x, y, z = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)
    
    # Calculate centrality for sizing
    centrality = G.degree(node)
    node_size.append(centrality * 2) 
    node_text.append(f"Node: {node}<br>Degree: {centrality}")

# Create the 3D Scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    marker=dict(
        size=node_size,
        color=node_size,
        colorscale='Viridis',
        opacity=0.8
    ),
    text=node_text,
    hoverinfo='text'
)])

fig.update_layout(
    title="Chrysalis: 3D Mythic Hub Dominance",
    scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Save as interactive HTML
fig.write_html("docs/3d_hub_visualization.html")
print("Interactive 3D visualization saved to docs/3d_hub_visualization.html")
