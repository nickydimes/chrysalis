import sqlite3
import pandas as pd
import argparse
from pathlib import Path


def get_db_connection(db_path: Path):
    return sqlite3.connect(db_path)


def list_views(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
    return [row[0] for row in cursor.fetchall()]


def query_to_dataframe(conn, query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Run SQL-native cross-modal analytics on Chrysalis data."
    )
    parser.add_argument("--query", type=str, help="Custom SQL query to run.")
    parser.add_argument(
        "--view", type=str, help="Name of a pre-defined view to display."
    )
    parser.add_argument(
        "--list_views", action="store_true", help="List available analytics views."
    )
    parser.add_argument("--db_path", type=str, default="data/chrysalis.db")
    parser.add_argument("--output_csv", type=str, help="Save result to a CSV file.")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    db_path = project_root / args.db_path

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        exit(1)

    conn = get_db_connection(db_path)

    if args.list_views:
        views = list_views(conn)
        print("Available Analytics Views:")
        for v in views:
            print(f"  - {v}")
        exit(0)

    query = args.query
    if args.view:
        query = f"SELECT * FROM {args.view}"

    if query:
        print(f"Executing Query: {query}")
        df = query_to_dataframe(conn, query)

        if not df.empty:
            print("\n--- Query Results ---")
            print(df.to_string(index=False))

            if args.output_csv:
                output_path = Path(args.output_csv)
                df.to_csv(output_path, index=False)
                print(f"\nResult saved to: {output_path}")
        else:
            print("No results returned.")
    else:
        print("Please specify a --query or --view. Use --list_views to see options.")

    conn.close()


if __name__ == "__main__":
    main()
