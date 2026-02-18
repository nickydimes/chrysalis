import json
from pathlib import Path
from typing import List, Dict, Any
from jsonschema import validate, ValidationError
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


def load_ethnographic_records(directory: Path, schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Loads and validates all JSON ethnographic records from a given directory
    and returns them as a Pandas DataFrame.
    """
    records_list = []
    invalid_files = []

    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                record = json.load(f)
            validate(instance=record, schema=schema)
            # Store filename for traceability
            record["_source_file"] = json_file.name
            records_list.append(record)
        except json.JSONDecodeError:
            invalid_files.append(f"{json_file.name} (Invalid JSON format)")
        except ValidationError as e:
            invalid_files.append(
                f"{json_file.name} (Schema validation failed: {e.message})"
            )
        except Exception as e:
            invalid_files.append(f"{json_file.name} (Unexpected error: {e})")

    if invalid_files:
        print("Warning: Some ethnographic records could not be loaded or validated:")
        for error_msg in invalid_files:
            print(f"  - {error_msg}")

    if not records_list:
        return pd.DataFrame()

    return pd.DataFrame(records_list)


def normalize_ethnographic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens nested JSON structures in the ethnographic DataFrame using pd.json_normalize.
    Specifically targets 'eight_step_protocol_relevance'.
    """
    if df.empty or "eight_step_protocol_relevance" not in df.columns:
        return df

    # Normalize the protocol relevance column
    # This creates columns like 'protocol.Purification', 'protocol.Dissolution', etc.
    protocol_df = pd.json_normalize(df["eight_step_protocol_relevance"])
    protocol_df.columns = [f"protocol.{col}" for col in protocol_df.columns]

    # Concatenate with the original DataFrame (dropping the original nested column)
    df_normalized = pd.concat(
        [df.drop(columns=["eight_step_protocol_relevance"]), protocol_df], axis=1
    )

    return df_normalized


def filter_records(df: pd.DataFrame, key: str, value: Any) -> pd.DataFrame:
    """
    Filters ethnographic records in a DataFrame. Supports nested keys if normalized.
    """
    if df.empty:
        return pd.DataFrame()

    if key == "tags":
        return df[
            df["tags"].apply(lambda x: value in (x if isinstance(x, list) else []))
        ]
    elif key in df.columns:
        return df[df[key] == value]
    else:
        print(f"Warning: Key '{key}' not found in DataFrame columns.")
        return df


def search_records(
    df: pd.DataFrame, keyword: str, fields: List[str] = None
) -> pd.DataFrame:
    """
    Searches for a keyword in specified text fields of ethnographic records.
    """
    if df.empty:
        return pd.DataFrame()
    if fields is None:
        # Default to title, summary, and all protocol mention fields if normalized
        fields = ["title", "summary"] + [
            col for col in df.columns if col.startswith("protocol.")
        ]

    mask = pd.Series([False] * len(df), index=df.index)
    for field in fields:
        if field in df.columns:
            mask |= df[field].astype(str).str.contains(keyword, case=False, na=False)

    return df[mask]


def aggregate_critical_elements(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates the counts of all critical elements observed across records.
    """
    if df.empty or "critical_elements_observed" not in df.columns:
        return pd.Series(dtype=int)

    return df["critical_elements_observed"].explode().value_counts()


def aggregate_protocol_relevance(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates how often each Eight-Step Protocol phase has valid relevance mentions.
    Works on both original and normalized DataFrames.
    """
    if df.empty:
        return pd.Series(dtype=int)

    protocol_steps = [
        "Purification",
        "Containment",
        "Anchoring",
        "Dissolution",
        "Liminality",
        "Encounter",
        "Integration",
        "Emergence",
    ]

    counts = {}

    # Check if normalized
    protocol_cols = [col for col in df.columns if col.startswith("protocol.")]
    if protocol_cols:
        for step in protocol_steps:
            col_name = f"protocol.{step}"
            if col_name in df.columns:
                # Count non-empty, non-default strings
                valid_mentions = df[col_name].astype(str).str.strip().str.lower()
                counts[step] = len(
                    df[
                        (valid_mentions != "")
                        & (valid_mentions != "not explicitly observed")
                        & (valid_mentions != "nan")
                    ]
                )
    else:
        # Fallback to original nested structure
        relevance_counts = Counter()
        for _, row in df.iterrows():
            relevance = row.get("eight_step_protocol_relevance", {})
            for step in protocol_steps:
                val = str(relevance.get(step, "")).strip().lower()
                if val and val != "not explicitly observed":
                    relevance_counts[step] += 1
        counts = dict(relevance_counts)

    return pd.Series(counts).sort_values(ascending=False)


def get_protocol_mentions(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the records and specific text mentions for a given protocol step.
    Requires a normalized DataFrame.
    """
    col_name = f"protocol.{step}"
    if col_name not in df.columns:
        print(f"Error: Step '{step}' not found. Ensure DataFrame is normalized.")
        return pd.DataFrame()

    valid_mask = (
        (df[col_name].astype(str).str.strip().str.lower() != "")
        & (
            df[col_name].astype(str).str.strip().str.lower()
            != "not explicitly observed"
        )
        & (df[col_name].astype(str) != "nan")
    )

    return df[valid_mask][["title", col_name, "_source_file"]]


def plot_bar_chart(
    data: pd.Series, title: str, xlabel: str, ylabel: str, filename: Path
):
    """
    Generates and saves a bar chart from a Pandas Series.
    """
    if data.empty:
        print(f"No data to plot for '{title}'. Skipping.")
        return

    plt.figure(figsize=(12, 7))
    data.plot(kind="bar", color="teal", alpha=0.7)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved plot to {filename}")


def main():
    print("--- Advanced Ethnographic Data Analysis (Pandas Refactor) ---")

    data_dir = Path("chrysalis/data/ethnographic")
    schema_path = Path("chrysalis/schema/ethnographic_record.json")
    output_plots_dir = Path("chrysalis/data/analysis_plots")
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists() or not schema_path.exists():
        print("Error: Missing data or schema.")
        exit(1)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # 1. Load records
    df = load_ethnographic_records(data_dir, schema)
    if df.empty:
        print("No records found.")
        exit(0)

    # 2. Normalize DataFrame (Flattening nested protocol relevance)
    df_norm = normalize_ethnographic_dataframe(df)
    print(f"\nNormalized DataFrame Columns:\n{df_norm.columns.tolist()}")

    # 3. Advanced Filtering & Searching
    # Search for 'identity' across all fields including protocol mentions
    identity_related = search_records(df_norm, "identity")
    print(f"\nFound {len(identity_related)} records mentioning 'identity'.")

    # 4. Deep-dive into specific protocol steps
    print("\nMentions for 'Dissolution':")
    dissolution_mentions = get_protocol_mentions(df_norm, "Dissolution")
    if not dissolution_mentions.empty:
        print(dissolution_mentions)

    # 5. Aggregations & Visualization
    print("\nAggregating Protocol Relevance...")
    protocol_counts = aggregate_protocol_relevance(df_norm)
    print(protocol_counts)
    plot_bar_chart(
        protocol_counts,
        "Mentions across Eight-Step Protocol",
        "Protocol Phase",
        "Count",
        output_plots_dir / "protocol_relevance_adv.png",
    )

    print("\nAggregating Critical Elements...")
    element_counts = aggregate_critical_elements(df_norm)
    plot_bar_chart(
        element_counts.head(15),
        "Top 15 Critical Elements Observed",
        "Element",
        "Frequency",
        output_plots_dir / "critical_elements_adv.png",
    )


if __name__ == "__main__":
    main()
