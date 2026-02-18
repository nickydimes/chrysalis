import sqlite3
import pytest
import networkx as nx
from chrysalis.src.tools.extract_knowledge_graph import (
    save_triples_to_db,
    build_networkx_graph,
)


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE knowledge_triples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        subject_type TEXT,
        predicate TEXT,
        object TEXT,
        object_type TEXT,
        source_file TEXT,
        extraction_date TEXT
    )
    """
    )
    conn.commit()
    yield conn
    conn.close()


def test_save_triples_to_db(memory_db):
    triples = [
        {
            "subject": "Ising Model",
            "subject_type": "SimulationModel",
            "predicate": "maps to",
            "object": "Dissolution",
            "object_type": "ProtocolStep",
        },
        {
            "subject": "Purification",
            "subject_type": "ProtocolStep",
            "predicate": "precedes",
            "object": "Containment",
            "object_type": "ProtocolStep",
        },
    ]
    save_triples_to_db(memory_db, triples, "test_file.json")

    cursor = memory_db.cursor()
    cursor.execute("SELECT count(*) FROM knowledge_triples")
    count = cursor.fetchone()[0]
    assert count == 2

    cursor.execute(
        "SELECT subject, object FROM knowledge_triples WHERE subject = 'Ising Model'"
    )
    row = cursor.fetchone()
    assert row[0] == "Ising Model"
    assert row[1] == "Dissolution"


def test_build_networkx_graph(memory_db):
    triples = [
        {
            "subject": "Node A",
            "subject_type": "Type X",
            "predicate": "connects to",
            "object": "Node B",
            "object_type": "Type Y",
        }
    ]
    save_triples_to_db(memory_db, triples, "test_file.json")

    G = build_networkx_graph(memory_db)

    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert G.nodes["Node A"]["type"] == "Type X"
    assert G.nodes["Node B"]["type"] == "Type Y"

    # Check edge data
    edge_data = list(G.edges(data=True))[0]
    assert edge_data[0] == "Node A"
    assert edge_data[1] == "Node B"
    assert edge_data[2]["predicate"] == "connects to"
