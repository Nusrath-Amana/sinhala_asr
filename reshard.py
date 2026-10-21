"""Re-shard stratified_v7 into parquet files a streaming reader can actually use. Run ONCE.

THE PROBLEM. Every split of this dataset is a single parquet row group:

    train       14.11 GB   108,278 rows   1 row group   (audio 14.1 GB, text 5.3 MB)
    validation   1.46 GB    14,084 rows   1 row group
    test         1.63 GB    15,860 rows   1 row group

A row group is the unit a parquet reader materialises. There is no way to take a slice of one, so
`load_dataset(..., streaming=True)` on `train` tries to pull 14 GB before yielding a single row
and Kaggle kills the session. It is the same reason the HF dataset viewer reports "Scan size limit
exceeded" on this repo. Column pruning rescues anything that needs only `text` (see vocab.py), but
training needs the audio.

THE FIX. Download once, rewrite with sane row groups, and stream from the shards forever after.

    python reshard.py --split validation --out /kaggle/temp/v7      # 1.5 GB, try this first
    python reshard.py --split train      --out /kaggle/temp/v7      # 14 GB, CPU session only

Run it on a **CPU session** (~30 GB RAM); a GPU session has far less and will die on `train`.
Then publish `--out` as a Kaggle Dataset, or keep it in /kaggle/temp for a single session's use,
and point train.py at it with `--repo /kaggle/input/<your-dataset>`.

Disk: the download plus the shards is roughly 2x the split's size, so budget ~30 GB for `train`.
Pass --no-keep-download to delete the source file as soon as the shards are written.
"""
from __future__ import annotations

import argparse
import os
import time

SPLITS = ("train", "validation", "test")
COLS = ["audio", "text", "source_dataset", "language"]


def human(n: float) -> str:
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7")
    ap.add_argument("--split", default="validation", choices=SPLITS)
    ap.add_argument("--out", default="shards")
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument("--row-group-size", type=int, default=200,
                    help="rows per row group INSIDE each shard -- this is the number that makes "
                         "streaming cheap; 200 rows is roughly 25 MB of audio")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--cache-dir", default=None, help="where to put the downloaded source file")
    ap.add_argument("--no-keep-download", action="store_true")
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(a.out, exist_ok=True)
    print(f"downloading {a.split}.parquet from {a.repo} ...", flush=True)
    t0 = time.time()
    src = hf_hub_download(repo_id=a.repo, filename=f"{a.split}.parquet",
                          repo_type="dataset", cache_dir=a.cache_dir)
    print(f"  {human(os.path.getsize(src))} in {time.time()-t0:.0f}s -> {src}", flush=True)

    # memory_map keeps the page cache, not the heap, holding the bytes; this is what makes
    # iter_batches viable on a single giant row group locally, where it is not over HTTP.
    pf = pq.ParquetFile(src, memory_map=True, pre_buffer=False)
    total = pf.metadata.num_rows
    print(f"  {total:,} rows, {pf.metadata.num_row_groups} row group(s) -> resharding", flush=True)

    writer = None
    shard = rows_in_shard = seen = 0
    t0 = time.time()

    def open_shard(schema):
        nonlocal writer, shard, rows_in_shard
        if writer is not None:
            writer.close()
        path = os.path.join(a.out, f"{a.split}-{shard:04d}.parquet")
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        shard += 1
        rows_in_shard = 0
        return path

    try:
        for batch in pf.iter_batches(batch_size=a.batch_size, columns=COLS):
            tbl = pa.Table.from_batches([batch])
            if writer is None or rows_in_shard >= a.rows_per_shard:
                open_shard(tbl.schema)
            writer.write_table(tbl, row_group_size=a.row_group_size)
            rows_in_shard += batch.num_rows
            seen += batch.num_rows
            if seen % (a.batch_size * 25) == 0:
                el = time.time() - t0
                print(f"  {seen:,}/{total:,} rows  {el:.0f}s  "
                      f"{seen/max(el,1e-9):.0f} rows/s  shards={shard}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    out_bytes = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)
                    if f.startswith(a.split) and f.endswith(".parquet"))
    print(f"\nwrote {shard} shards, {seen:,} rows, {human(out_bytes)} to {a.out}")
    if seen != total:
        print(f"  WARNING: expected {total:,} rows, wrote {seen:,}")

    if a.no_keep_download:
        try:
            os.remove(src)
            print(f"  removed the source download ({src})")
        except OSError as e:
            print(f"  could not remove {src}: {e}")

    print("\nNext: point training at the shards, e.g.")
    print(f"  python train.py --repo {a.out} --freeze-encoder --max-steps 200 --subset 500")


if __name__ == "__main__":
    main()
