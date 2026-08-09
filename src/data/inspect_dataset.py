"""
Dataset Inspection Script
=========================

Verifies Hugging Face authentication, dataset access, and streaming
functionality. Prints schema information and sample records.

Usage:
    python src/data/inspect.py
"""

import sys
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME: str = "GD-Studio/embeat_45m_spotify_tracks"
DATASET_SPLIT: str = "train"
NUM_SAMPLE_RECORDS: int = 5


def inspect_dataset() -> None:
    """Connect to the Hugging Face dataset, stream a few records, and print
    schema information plus sample data."""

    # ------------------------------------------------------------------
    # 1. Import & authenticate
    # ------------------------------------------------------------------
    print("=" * 60)
    print("VNYL Music Recommendation — Dataset Inspector")
    print("=" * 60)
    print()

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: The 'datasets' package is not installed.")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Attempt to load the dataset in streaming mode
    # ------------------------------------------------------------------
    print(f"Connecting to dataset: {DATASET_NAME}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Mode: streaming")
    print()

    try:
        dataset = load_dataset(
            DATASET_NAME,
            split=DATASET_SPLIT,
            streaming=True,
        )
    except Exception as exc:
        _handle_connection_error(exc)
        sys.exit(1)

    print("✓ Dataset connection successful")
    print()

    # ------------------------------------------------------------------
    # 3. Read a small sample of records
    # ------------------------------------------------------------------
    print(f"Reading first {NUM_SAMPLE_RECORDS} records …")
    print()

    records: list[dict[str, Any]] = []
    try:
        for i, record in enumerate(dataset):
            if i >= NUM_SAMPLE_RECORDS:
                break
            records.append(record)
    except Exception as exc:
        _handle_connection_error(exc)
        sys.exit(1)

    if not records:
        print("WARNING: No records were returned by the dataset.")
        sys.exit(1)

    print(f"Number of inspected records: {len(records)}")
    print()

    # ------------------------------------------------------------------
    # 4. Print available fields / schema
    # ------------------------------------------------------------------
    fields = list(records[0].keys())
    print(f"Available fields ({len(fields)}):")
    print("-" * 40)
    for field in fields:
        sample_value = records[0][field]
        value_type = type(sample_value).__name__
        print(f"  {field:<30s}  ({value_type})")
    print()

    # ------------------------------------------------------------------
    # 5. Print sample records
    # ------------------------------------------------------------------
    print("Sample records:")
    print("-" * 60)
    for idx, record in enumerate(records):
        print(f"\n--- Record {idx + 1} ---")
        for key, value in record.items():
            # Truncate very long values for readability
            display_value = str(value)
            if len(display_value) > 120:
                display_value = display_value[:117] + "..."
            print(f"  {key:<30s}: {display_value}")
    print()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Inspection complete.")
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Fields:           {len(fields)}")
    print(f"  Records sampled:  {len(records)}")
    print("=" * 60)


def _handle_connection_error(exc: Exception) -> None:
    """Print a user-friendly error message for common failures."""
    error_msg = str(exc).lower()

    if "401" in error_msg or "unauthorized" in error_msg or "token" in error_msg:
        print("ERROR: Hugging Face authentication failed.")
        print()
        print("Please authenticate by running:")
        print("  hf auth login")
        print()
        print("You need a Hugging Face account and a valid access token.")
        print("Create a token at: https://huggingface.co/settings/tokens")
    elif "403" in error_msg or "forbidden" in error_msg or "gated" in error_msg:
        print("ERROR: Access denied to the dataset.")
        print()
        print("This dataset may require you to accept access conditions.")
        print("Please visit the dataset page and accept the terms:")
        print(f"  https://huggingface.co/datasets/{DATASET_NAME}")
        print()
        print("After accepting, retry this script.")
    elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
        print("ERROR: Network connection failed.")
        print()
        print("Please check your internet connection and try again.")
    else:
        print(f"ERROR: Failed to access the dataset.")
        print(f"  Details: {exc}")
        print()
        print("Troubleshooting:")
        print("  1. Run: hf auth login")
        print(f"  2. Visit: https://huggingface.co/datasets/{DATASET_NAME}")
        print("  3. Accept dataset access conditions if required")
        print("  4. Check your internet connection")


if __name__ == "__main__":
    inspect_dataset()
