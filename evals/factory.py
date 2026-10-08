import os
from typing import Dict

from evals.base import BaseBenchmark
from evals.blink import BlinkBench
from evals.mindcube import MindCubeBench
from evals.mmsi import MMSIBench
from evals.omnispatial import OmniSpatialBench
from evals.spinbench import SpinBench


BENCHMARK_REGISTRY: Dict[str, BaseBenchmark] = {
    'mindcube': MindCubeBench,
    'mmsi': MMSIBench,
    'omnispatial': OmniSpatialBench,
    'spinbench': SpinBench,
    'blink': BlinkBench,
    'none': None
}


class BenchmarkFactory:

    @staticmethod
    def create_benchmark(benchmark_name: str, **kwargs) -> BaseBenchmark:
        benchmark_class = BENCHMARK_REGISTRY.get(benchmark_name.lower())
        if not benchmark_class:
            raise ValueError(
                f'Unknown benchmark: "{benchmark_name}". '
                f'Available benchmarks: {list(BENCHMARK_REGISTRY.keys())}'
            )
        
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', benchmark_name.lower()
        )
        return benchmark_class(data_path=data_path, **kwargs)
