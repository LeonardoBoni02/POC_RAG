from datasets import load_dataset
dataset=load_dataset("nvidia/ChatRAG_Bench", "doc2dial")

from datasets import load_dataset
import pandas as pd
import os
import csv

# max file size per CSV in bytes (25 MB)
MAX_BYTES = 25 * 1024 * 1024

dataset = load_dataset("nvidia/ChatRAG-Bench", "doc2dial")

out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)


def write_limited_csv(df: pd.DataFrame, path: str, max_bytes: int = MAX_BYTES, rows_limit: int | None = None) -> int:
    """Write DataFrame to CSV row-by-row and stop when file size would exceed max_bytes.

    Returns number of rows written.
    """
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    written = 0
    # Use DictWriter to write header then rows
    with open(path, "w", newline="", encoding="utf-8") as f:
        if df.empty:
            f.write("")
            return 0
        fieldnames = list(df.columns)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        # If header alone already exceeds limit, truncate file to header only
        if os.path.getsize(path) >= max_bytes:
            return 0

        # Iterate rows and write until size limit reached or rows_limit hit
        for i, row in enumerate(df.to_dict(orient="records")):
            writer.writerow(row)
            written += 1
            # stop if we've reached a row limit
            if rows_limit is not None and written >= rows_limit:
                break
            # check size periodically (every 100 rows) to reduce syscall overhead
            if written % 100 == 0:
                f.flush()
                if os.path.getsize(path) >= max_bytes:
                    # we stopped shortly after exceeding the limit
                    break
        # final size check
        f.flush()
    # If file exceeds max_bytes, it's acceptable but we stopped soon after reaching it.
    return written


# Save each HF split as CSV (limited to MAX_BYTES)
for split_name, ds in dataset.items():  # dataset is a DatasetDict
    try:
        n = len(ds)
    except TypeError:
        # streaming IterableDataset may not support len()
        n = None
    print(f"Processing split: {split_name}, rows: {n if n is not None else 'unknown (streaming)'}")
    df = ds.to_pandas()
    out_path = os.path.join(out_dir, f"{split_name}.csv")
    rows_written = write_limited_csv(df, out_path, max_bytes=MAX_BYTES)
    final_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"Saved {out_path} ({rows_written} rows, {final_size / 1024:.1f} KB)")


# Create derived train/validation/test splits using a source DataFrame.
# If the original dataset has a 'train' split we'll use it; otherwise we
# fall back to a single split (if only one exists) or concatenate all splits.
if "train" in dataset:
    source_df = dataset["train"].to_pandas()
    source_name = "train"
else:
    splits = list(dataset.keys())
    if len(splits) == 1:
        source_df = dataset[splits[0]].to_pandas()
        source_name = splits[0]
    else:
        # concatenate all splits into one DataFrame
        dfs = [dataset[k].to_pandas() for k in splits]
        source_df = pd.concat(dfs, ignore_index=True)
        source_name = "combined_" + "_".join(splits)

# Shuffle for a random split (reproducible)
source_df = source_df.sample(frac=1, random_state=42).reset_index(drop=True)
# We want derived splits in proportions: 50% train, 25% val, 25% test
TRAIN_FRAC = 0.50
VAL_FRAC = 0.25
TEST_FRAC = 0.25


def estimate_avg_row_size(df: pd.DataFrame, sample_n: int = 200) -> float:
    """Estimate average bytes per CSV row (excluding header) by sampling.

    Falls back to a conservative estimate if df is empty.
    """
    if df.empty:
        return 0.0
    sample = df.head(sample_n).to_dict(orient="records")
    # write sample rows to a string buffer using csv to include quoting/escaping
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(df.columns))
    writer.writeheader()
    for row in sample:
        writer.writerow(row)
    s = buf.getvalue()
    # bytes of rows only (exclude header roughly)
    header_end = s.find("\n")
    if header_end == -1:
        return max(1.0, len(s) / max(1, len(sample)))
    rows_bytes = len(s.encode("utf-8")) - len(s[: header_end + 1].encode("utf-8"))
    avg = rows_bytes / max(1, len(sample))
    return avg


# Build initial splits indexes according to proportions
n = len(source_df)
train_size = int(TRAIN_FRAC * n)
remaining = n - train_size
validation_size = remaining // 2
test_size = remaining - validation_size

train_split = source_df.iloc[:train_size]
validation_split = source_df.iloc[train_size:train_size + validation_size]
test_split = source_df.iloc[train_size + validation_size:]

# Estimate average bytes per row for each split
avg_train = estimate_avg_row_size(train_split)
avg_val = estimate_avg_row_size(validation_split)
avg_test = estimate_avg_row_size(test_split)

# Compute maximum rows that fit per-split given MAX_BYTES
def max_rows_by_size(avg_row_bytes: float) -> int:
    if avg_row_bytes <= 0:
        return 0
    return max(0, int(MAX_BYTES // avg_row_bytes))

max_train_rows = max_rows_by_size(avg_train)
max_val_rows = max_rows_by_size(avg_val)
max_test_rows = max_rows_by_size(avg_test)

# Now compute a global k so that saved rows keep the 50/25/25 proportions and do not exceed per-split maxima
# We need: 0.5*k <= max_train_rows, 0.25*k <= max_val_rows, 0.25*k <= max_test_rows
import math

limits = []
if max_train_rows > 0:
    limits.append(max_train_rows / TRAIN_FRAC)
if max_val_rows > 0:
    limits.append(max_val_rows / VAL_FRAC)
if max_test_rows > 0:
    limits.append(max_test_rows / TEST_FRAC)

# Also ensure we don't request more rows than available in each split
if train_size > 0:
    limits.append(train_size / TRAIN_FRAC)
if validation_size > 0:
    limits.append(validation_size / VAL_FRAC)
if test_size > 0:
    limits.append(test_size / TEST_FRAC)

if limits:
    k = int(math.floor(min(limits)))
else:
    k = 0

# k is the total 'units' corresponding to the proportions; now compute counts
train_count = int(round(TRAIN_FRAC * k))
val_count = int(round(VAL_FRAC * k))
test_count = k - train_count - val_count

# Ensure counts do not exceed available rows
train_count = min(train_count, len(train_split))
val_count = min(val_count, len(validation_split))
test_count = min(test_count, len(test_split))

# If all counts are zero (e.g., tiny MAX_BYTES), fall back to writing at least one row per split if available
if train_count == 0 and len(train_split) > 0:
    train_count = 1
if val_count == 0 and len(validation_split) > 0:
    val_count = 1
if test_count == 0 and len(test_split) > 0:
    test_count = 1

# Finally write the limited CSVs using the computed row counts (also still enforced by MAX_BYTES)
train_rows = write_limited_csv(train_split, os.path.join(out_dir, "train.csv"), max_bytes=MAX_BYTES, rows_limit=train_count)
val_rows = write_limited_csv(validation_split, os.path.join(out_dir, "validation.csv"), max_bytes=MAX_BYTES, rows_limit=val_count)
test_rows = write_limited_csv(test_split, os.path.join(out_dir, "test.csv"), max_bytes=MAX_BYTES, rows_limit=test_count)

train_size_bytes = os.path.getsize(os.path.join(out_dir, "train.csv")) if os.path.exists(os.path.join(out_dir, "train.csv")) else 0
val_size_bytes = os.path.getsize(os.path.join(out_dir, "validation.csv")) if os.path.exists(os.path.join(out_dir, "validation.csv")) else 0
test_size_bytes = os.path.getsize(os.path.join(out_dir, "test.csv")) if os.path.exists(os.path.join(out_dir, "test.csv")) else 0

print(
    f"Derived splits saved from '{source_name}': train={train_rows} rows ({train_size_bytes/1024:.1f} KB), validation={val_rows} rows ({val_size_bytes/1024:.1f} KB), test={test_rows} rows ({test_size_bytes/1024:.1f} KB)"
)
