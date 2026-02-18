import argparse
import sqlite3
from pathlib import Path
import networkx as nx


def build_networkx_graph(db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT subject, subject_type, predicate, object, object_type FROM knowledge_triples"
    )
    rows = cursor.fetchall()

    G = nx.MultiDiGraph()
    for row in rows:
        subj, subj_type, pred, obj, obj_type = row
        G.add_node(subj, type=subj_type)
        G.add_node(obj, type=obj_type)
        G.add_edge(subj, obj, predicate=pred)
    conn.close()
    return G


def find_hidden_relationships(G):
    """
    Looks for non-trivial paths between ethnographic entities and physical models.
    Example: Event -> Observation -> ProtocolPhase -> SimModel
    """
    discoveries = []

    # Find all 'Event' nodes and 'SimModel' nodes
    events = [n for n, d in G.nodes(data=True) if d.get("type") == "Event"]
    models = [n for n, d in G.nodes(data=True) if d.get("type") == "SimModel"]

    for event in events:
        for model in models:
            try:
                # Find the shortest path between an event and a model
                paths = list(
                    nx.all_simple_paths(G, source=event, target=model, cutoff=4)
                )
                if paths:
                    discoveries.append(
                        {"source": event, "target": model, "paths": paths}
                    )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    return discoveries


def main():
    parser = argparse.ArgumentParser(description="Query the Chrysalis Knowledge Graph.")
    parser.add_argument("--node", type=str, help="Get neighbors of a specific node.")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Find hidden relationships (Event -> Model paths).",
    )
    parser.add_argument("--db_path", type=str, default="data/chrysalis.db")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    db_path = project_root / args.db_path

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        exit(1)

    G = build_networkx_graph(db_path)
    print(
        f"Loaded Knowledge Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    if args.node:
        print(f"\nNeighbors of '{args.node}':")
        if args.node in G:
            for neighbor in G.neighbors(args.node):
                edges = G.get_edge_data(args.node, neighbor)
                for idx, edge_data in edges.items():
                    print(
                        f"  --({edge_data['predicate']})--> {neighbor} ({G.nodes[neighbor].get('type')})"
                    )
        else:
            print(f"Node '{args.node}' not found.")

    if args.discover:
        print("\n--- Hidden Relationship Discovery (Event to Simulation Model) ---")
        discoveries = find_hidden_relationships(G)
        if not discoveries:
            print("No cross-modal paths found yet. Try extracting more data.")
        else:
            for d in discoveries:
                print(
                    f"Discovery: Event '{d['source']}' connects to SimModel '{d['target']}'"
                )
                for i, path in enumerate(d["paths"]):
                    path_str = " -> ".join(
                        [f"{p} ({G.nodes[p].get('type')})" for p in path]
                    )
                    print(f"  Path {i+1}: {path_str}")


if __name__ == "__main__":
    main()
