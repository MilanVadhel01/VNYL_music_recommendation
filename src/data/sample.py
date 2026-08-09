"""
1M Song Sampler
===============

Streams records from the Hugging Face dataset and writes them
incrementally to a Parquet file using PyArrow's ParquetWriter.

Records are processed in chunks of CHUNK_SIZE to avoid excessive
memory usage. After writing, the script validates the resulting
Parquet file.

Usage:
    python src/data/sample.py

Notes:
    - The first working version reads records sequentially.
    - The sampling strategy (sequential vs. random/deterministic)
      can be swapped by replacing the `_stream_records` function.
    - Checkpointing is not implemented in this version to keep
      complexity low. If the script is interrupted, delete the
      partial output file and re-run.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME: str = "GD-Studio/embeat_45m_spotify_tracks"
DATASET_SPLIT: str = "train"
TARGET_ROWS: int = 1_000_000
CHUNK_SIZE: int = 10_000
OUTPUT_PATH: str = os.path.join("data", "processed", "songs_1m.parquet")
PROGRESS_INTERVAL: int = 10_000  # Print progress every N rows


# ---------------------------------------------------------------------------
# Streaming & collection
# ---------------------------------------------------------------------------
def _stream_records(
    dataset_name: str,
    split: str,
    target: int,
) -> Iterator[list[dict[str, Any]]]:
    """Yield chunks of records from the streaming dataset.

    Each yielded list contains up to CHUNK_SIZE records.
    Stops after *target* total records have been yielded.
    """
    from datasets import load_dataset  # type: ignore

    print(f"Connecting to dataset: {dataset_name} …")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    print("✓ Connected\n")

    chunk: list[dict[str, Any]] = []
    collected = 0

    for record in dataset:
        if collected >= target:
            break

        chunk.append(record)
        collected += 1

        if len(chunk) >= CHUNK_SIZE:
            yield chunk
            chunk = []

        if collected % PROGRESS_INTERVAL == 0:
            print(f"  Collected: {collected:>10,d} / {target:,d}")

    # Yield any remaining records in the last partial chunk
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------
def _write_parquet(
    output_path: str,
    dataset_name: str,
    split: str,
    target: int,
) -> int:
    """Stream records in chunks, convert to Arrow tables, and write
    incrementally to a single Parquet file. Returns the number of
    rows written."""

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_written = 0

    try:
        for chunk in _stream_records(dataset_name, split, target):
            table = pa.Table.from_pylist(chunk)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)
            total_written += len(chunk)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")
        print(f"  Rows written so far: {total_written:,d}")
        print(f"  Partial file at: {output_path}")
        print("  Delete the file and re-run to start over.")
        sys.exit(1)

    except Exception as exc:
        _handle_stream_error(exc, total_written, output_path)
        sys.exit(1)

    finally:
        if writer is not None:
            writer.close()

    return total_written


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_parquet(output_path: str) -> None:
    """Load the Parquet file and print validation statistics."""

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    df = pd.read_parquet(output_path)

    # --- Row count ---
    print(f"\nRow count: {len(df):,d}")

    # --- Columns ---
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}")

    # --- Data types ---
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col:<35s}  {df[col].dtype}")

    # --- Duplicate track IDs ---
    if "track_id" in df.columns:
        dup_count = df["track_id"].duplicated().sum()
        print(f"\nDuplicate track_id values: {dup_count:,d}")
    else:
        print("\nNote: 'track_id' column not found — skipping duplicate check.")

    # --- Missing values ---
    missing = df.isnull().sum()
    print(f"\nMissing values:")
    for col in df.columns:
        print(f"  {col:<35s}  {missing[col]:>8,d}")

    # --- Numeric statistics ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\nNumeric feature statistics:")
        print(df[numeric_cols].describe().to_string())
    else:
        print("\nNo numeric columns detected for statistics.")

    # --- File size ---
    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"\nParquet file size: {file_size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def _handle_stream_error(
    exc: Exception, rows_written: int, output_path: str
) -> None:
    """Print a user-friendly error for streaming failures."""
    error_msg = str(exc).lower()

    print(f"\n\nERROR: Failed during data extraction.")
    print(f"  Rows written before failure: {rows_written:,d}")

    if "401" in error_msg or "unauthorized" in error_msg or "token" in error_msg:
        print("\n  Cause: Hugging Face authentication failed.")
        print("  Fix:   Run  hf auth login")
    elif "403" in error_msg or "forbidden" in error_msg or "gated" in error_msg:
        print("\n  Cause: Dataset access denied.")
        print(f"  Fix:   Visit https://huggingface.co/datasets/{DATASET_NAME}")
        print("         and accept the access conditions.")
    elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
        print("\n  Cause: Network error.")
        print("  Fix:   Check your internet connection and retry.")
    else:
        print(f"\n  Details: {exc}")

    if rows_written > 0:
        print(f"\n  A partial file may exist at: {output_path}")
        print("  Delete it and re-run to start fresh.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: extract 1M songs and validate the output."""

    print("=" * 60)
    print("VNYL Music Recommendation — 1M Song Sampler")
    print("=" * 60)
    print()
    print(f"  Dataset:     {DATASET_NAME}")
    print(f"  Target rows: {TARGET_ROWS:,d}")
    print(f"  Chunk size:  {CHUNK_SIZE:,d}")
    print(f"  Output:      {OUTPUT_PATH}")
    print()

    # Check if output already exists
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_parquet(OUTPUT_PATH)
        print(f"⚠ Output file already exists with {len(existing):,d} rows.")
        print(f"  Path: {OUTPUT_PATH}")
        response = input("  Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("  Skipping extraction. Running validation on existing file …")
            _validate_parquet(OUTPUT_PATH)
            return
        print()

    start_time = time.time()
    print("Starting dataset extraction …\n")

    total = _write_parquet(OUTPUT_PATH, DATASET_NAME, DATASET_SPLIT, TARGET_ROWS)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print(f"\n✓ Extraction complete.")
    print(f"  Rows written:  {total:,d}")
    print(f"  Time elapsed:  {minutes}m {seconds:.1f}s")
    print(f"  Output file:   {OUTPUT_PATH}")

    # Validate
    _validate_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    main()
