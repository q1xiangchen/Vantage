"""Extract the SPAR-Bench test split from its Hugging Face Parquet shards."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq


def prepare(shards, output):
    images = output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    metadata = output / 'sparbench_test.jsonl'
    temporary = metadata.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    row['images'] = []
                    for index, item in enumerate(row.pop('image')):
                        suffix = Path(item['path'] or '').suffix or '.jpg'
                        filename = f"{int(row['id']):05d}_{index}{suffix}"
                        (images / filename).write_bytes(item['bytes'])
                        row['images'].append(f'images/{filename}')
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet_dir', type=Path, help='Directory containing downloaded test shards.')
    parser.add_argument('--output', type=Path, default=Path('data/sparbench'))
    args = parser.parse_args()
    source = args.parquet_dir
    if source is None:
        snapshot = snapshot_download('jasonzhango/SPAR-Bench', repo_type='dataset',
                                     allow_patterns='data/test-*.parquet')
        source = Path(snapshot) / 'data'
    shards = sorted(source.glob('test-*.parquet'))
    if not shards:
        parser.error(f'no test Parquet shards found in {source}')
    prepare(shards, args.output)


if __name__ == '__main__':
    main()
