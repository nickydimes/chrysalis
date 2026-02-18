import argparse
import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import networkx as nx
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)


def get_db_connection(db_path: Path):
    return sqlite3.connect(db_path)


def save_triples_to_db(conn, triples: List[Dict[str, Any]], source_file: str):
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    for t in triples:
        cursor.execute(
            """
        INSERT INTO knowledge_triples (subject, subject_type, predicate, object, object_type, source_file, extraction_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                t["subject"],
                t["subject_type"],
                t["predicate"],
                t["object"],
                t["object_type"],
                source_file,
                now,
            ),
        )
    conn.commit()


def build_networkx_graph(conn):
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
    return G


def main():
    parser = argparse.ArgumentParser(
        description="Extract Knowledge Graph triples from project data using an LLM."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ethnographic",
        help="Directory with structured JSON records.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="simulations/reports",
        help="Directory with simulation reports.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/chrysalis.db",
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--llm_client", type=str, choices=["gemini_api", "ollama"], default="ollama"
    )
    parser.add_argument("--llm_model", type=str, default="deepseek-r1:32b")
    parser.add_argument(
        "--template_name", type=str, default="graph_extraction_template"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    data_dir = project_root / args.data_dir
    reports_dir = project_root / args.reports_dir
    db_path = project_root / args.db_path
    template_path = project_root / "prompts" / "templates" / f"{args.template_name}.md"

    if not template_path.exists():
        print(f"Error: Template file '{template_path}' not found.")
        exit(1)

    try:
        # 1. Initialize LLM Client
        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            llm_client = OllamaClient(model_name=args.llm_model)

        print(f"Using LLM client: {args.llm_client} (model: {llm_client.model_name})")

        conn = get_db_connection(db_path)

        # 2. Extract from Ethnographic JSONs
        if data_dir.exists():
            for json_file in data_dir.glob("*.json"):
                print(f"Extracting triples from: {json_file.name}")
                content = json_file.read_text(encoding="utf-8")

                template_content = template_path.read_text(encoding="utf-8")
                prompt = template_content.replace("{DATA_CONTENT}", content)

                response = llm_client.generate_text(prompt)

                # Extract JSON list from response
                json_match = re.search(r"(\[.*\])", response, re.DOTALL)
                if json_match:
                    try:
                        triples = json.loads(json_match.group(1))
                        save_triples_to_db(conn, triples, json_file.name)
                        print(f"  Extracted {len(triples)} triples.")
                    except json.JSONDecodeError:
                        print(
                            f"  Warning: Failed to parse LLM response as JSON list for {json_file.name}"
                        )
                else:
                    print(
                        f"  Warning: No JSON list found in LLM response for {json_file.name}"
                    )

        # 3. Extract from Simulation Reports (Interpretations)
        if reports_dir.exists():
            for interp_file in reports_dir.glob("**/interpretation.md"):
                print(
                    f"Extracting triples from: {interp_file.parent.name}/{interp_file.name}"
                )
                content = interp_file.read_text(encoding="utf-8")

                template_content = template_path.read_text(encoding="utf-8")
                prompt = template_content.replace("{DATA_CONTENT}", content)

                response = llm_client.generate_text(prompt)

                json_match = re.search(r"(\[.*\])", response, re.DOTALL)
                if json_match:
                    try:
                        triples = json.loads(json_match.group(1))
                        save_triples_to_db(
                            conn,
                            triples,
                            f"{interp_file.parent.name}/{interp_file.name}",
                        )
                        print(f"  Extracted {len(triples)} triples.")
                    except json.JSONDecodeError:
                        print(
                            f"  Warning: Failed to parse LLM response as JSON list for {interp_file.name}"
                        )
                else:
                    print(
                        f"  Warning: No JSON list found in LLM response for {interp_file.name}"
                    )

        # 4. Save NetworkX GraphML
        print("\nBuilding NetworkX graph...")
        G = build_networkx_graph(conn)
        graphml_path = project_root / "data" / "knowledge_graph.graphml"
        nx.write_graphml(G, str(graphml_path))
        print(f"Knowledge Graph saved to {graphml_path}")

        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
