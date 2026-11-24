"""Fixed sample selection and sampling-preserving resume decisions."""
import json
from pathlib import Path


def select_sample_ids(benchmark, path):
    if not path:
        return
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    ids = json.loads(path.read_text())
    if not isinstance(ids, list) or len(ids) != len(set(map(str, ids))):
        raise ValueError('sample_ids_file must contain a list of unique IDs')
    selected = set(map(str, ids))
    benchmark_ids = benchmark.sample_ids().map(str)
    available = set(benchmark_ids)
    if selected - available:
        raise ValueError(f'Sample IDs absent from benchmark selection: {sorted(selected - available)[:10]}')
    benchmark.data = benchmark.data[benchmark_ids.isin(selected)].reset_index(drop=True)


def completed_prediction(record):
    # A sampled empty/truncated answer is final. Retry only genuine failures.
    return record.get('status') in ('complete', 'truncated') or bool(record.get('content'))
