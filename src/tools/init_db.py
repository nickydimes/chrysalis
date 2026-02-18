import sqlite3
import argparse
from pathlib import Path

def init_db(db_path: Path):
    """Creates the Chrysalis SQLite database and tables."""
    print(f"Initializing database at: {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create experiments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        name TEXT,
        simulation_module TEXT,
        config_file TEXT,
        output_base_dir TEXT,
        start_time TEXT,
        end_time TEXT,
        status TEXT,
        git_commit TEXT,
        python_version TEXT,
        os_info TEXT,
        global_params_json TEXT
    )
    """)

    # 2. Create simulation_runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_runs (
        run_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        seed INTEGER,
        status TEXT,
        start_time TEXT,
        end_time TEXT,
        output_path TEXT,
        parameters_json TEXT,
        error_message TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
    )
    """)

    # 3. Create ethnographic_records table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ethnographic_records (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        source TEXT,
        period_or_event TEXT,
        geographic_location TEXT,
        summary TEXT,
        source_file TEXT,
        critical_elements_json TEXT,
        protocol_relevance_json TEXT,
        tags_json TEXT
    )
    """)

    # 4. Create discovery_state table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS discovery_state (
        file_path TEXT PRIMARY KEY,
        status TEXT, -- 'Pending', 'Ingested', 'Hypothesized', 'Simulated', 'Interpreted', 'Meta-Analyzed'
        last_updated TEXT,
        experiment_id TEXT, -- Link to the simulation experiment if applicable
        FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
    )
    """)

    # 5. Create knowledge_triples table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_triples (
        triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        subject_type TEXT,
        predicate TEXT,
        object TEXT,
        object_type TEXT,
        source_file TEXT,
        extraction_date TEXT
    )
    """)

    # --- VIEWS FOR ANALYTICS ---

    # View linking Ethnographic Records to Simulation Experiments
    # Handles filename mapping (txt -> json)
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS vw_cross_modal_results AS
    SELECT 
        er.title AS event_title,
        er.summary AS ethnographic_summary,
        er.critical_elements_json,
        ds.status AS discovery_status,
        ex.name AS experiment_name,
        ex.simulation_module,
        sr.run_id,
        sr.seed,
        sr.status AS run_status,
        sr.parameters_json
    FROM ethnographic_records er
    JOIN discovery_state ds ON 
        (er.source_file = REPLACE(REPLACE(ds.file_path, 'data/raw/', ''), '.txt', '.json'))
    JOIN experiments ex ON ds.experiment_id = ex.experiment_id
    JOIN simulation_runs sr ON ex.experiment_id = sr.experiment_id
    """)

    # View showing Protocol Throughput
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS vw_protocol_throughput AS
    SELECT 
        status, 
        COUNT(*) as count,
        STRFTIME('%Y-%m-%d', last_updated) as date
    FROM discovery_state
    GROUP BY status, date
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def main():
    parser = argparse.ArgumentParser(description="Initialize the Chrysalis SQLite database.")
    parser.add_argument("--db_path", type=str, default="chrysalis/data/chrysalis.db",
                        help="Path to the SQLite database file.")
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)

if __name__ == "__main__":
    main()
