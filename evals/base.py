from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class BaseBenchmarkSample:
    sample_id: int | str
    question: str
    question_type: str
    images: List[str]
    answer: str


class BaseBenchmark:
    data_specific_prompt: str = ''
    # The benchmark's own official answer-format wording (from its reference eval
    # harness), selectable via apply_prompt_style().
    official_data_specific_prompt: Optional[str] = None

    def __init__(self):
        self.n = 0

    def apply_prompt_style(self, style: str) -> None:
        """Switch the final-VQA answer format to the
        benchmark's official wording ('official').
        by benchmarks that override this (mindcube); everywhere else it falls
        back to 'official'. 'default' is a no-op."""
        if style == 'default':
            return
        if self.official_data_specific_prompt is not None:
            self.data_specific_prompt = self.official_data_specific_prompt

    def __getitem__(self, index: int) -> BaseBenchmarkSample:
        pass

    def __len__(self) -> int:
        pass

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self) -> BaseBenchmarkSample:
        if self.n < len(self):
            sample = self[self.n]
            self.n += 1
            return sample
        else:
            raise StopIteration

    def extract_answer(self, prediction: str) -> Optional[str]:
        pass

    def evaluate(
        self, 
        predictions: List[str] | Dict[int, str],
        output_dir: str = None,
        ignore_empty: bool = False,
    ):
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: list aligned with data rows, or dict {id: prediction}
            output_dir: optional path to save detailed results
        
        Returns:
            Dict with overall and per-type accuracy and detailed rows
        """
        pass

    def pretty_print_results(results: Dict, output_dir: Optional[str] = None):
        pass

    def save_results(self, results: Dict, output_file: str):
        detailed_df = pd.DataFrame(results['detailed_results'])
        if output_file.endswith('.xlsx'):
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                detailed_df.to_excel(writer, sheet_name='Details', index=False)
        elif output_file.endswith('.csv'):
            detailed_df.to_csv(output_file, index=False)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        # Enrich predictions.jsonl with per-sample question_type + is_correct so the
        # offline reader has everything it needs there and can skip results.csv
        # (whose full `prediction` column is huge on baseline runs). Matches by
        # sample_id; samples missing from detailed_results (e.g. dropped by
        # ignore_empty) are left untouched.
        self._enrich_predictions_jsonl(
            os.path.join(os.path.dirname(output_file), 'predictions.jsonl'),
            results['detailed_results'])

    @staticmethod
    def _enrich_predictions_jsonl(path: str, detailed_results: List[Dict]) -> None:
        if not os.path.exists(path):
            return
        scored = {str(r['id']): r for r in detailed_results}
        rows: List[Dict] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sr = scored.get(str(rec.get('sample_id')))
                if sr is not None:
                    rec['question_type'] = sr.get('question_type')
                    rec['is_correct'] = bool(sr.get('is_correct'))
                rows.append(rec)
        with open(path, 'w', encoding='utf-8') as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
