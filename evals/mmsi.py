from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from evals.base import BaseBenchmark, BaseBenchmarkSample


def extract_letter(text: Optional[str], letters: str = 'ABCD') -> Optional[str]:
    """Return the option letter this prediction most plausibly encodes, or None.

    MMSI is scored with this permissive parser rather than the benchmark's strict
    one: under the official prompt (letter wrapped in backticks) several models
    wrap or restate the answer in ways the strict parser rejects even though the
    chosen option is unambiguous. Priority order (first match wins; last
    occurrence within each pattern except boxed/GLM/backtick):

      1. \\boxed{X}   \\boxed{\\text{X}}   \\boxed{\\mathrm{X}}
      2. <|begin_of_box|>X<|end_of_box|>   |   <answer>...X...</answer>
      3. A. / B. (last match)              |   `X` (MMSI backtick)
      4. "answer is X" / "my answer is X" / "Answer: X" / "option X" (last match)
      5. (A) / A:                                                    (last match)
      6. \\b[A-D]\\b                                              (last letter)
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    lr = f'[{letters}]'

    # 1. \boxed variants (letter directly, or wrapped in \text{}/\mathrm{})
    for pat in (
        rf'\\boxed\{{\s*\\text\{{\s*({lr})\.?\s*\}}\s*\}}',
        rf'\\boxed\{{\s*\\mathrm\{{\s*({lr})\.?\s*\}}\s*\}}',
        rf'\\boxed\{{\s*({lr})\.?\s*\}}',
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2. GLM-4.5V box  |  <answer> / <Answer> tags — same tier.
    # For the tag branch, take the last STANDALONE uppercase A-X inside the
    # last tag. Word-boundary (\b) rejects letters embedded in words
    # ("Boxes"->B, "Cabinet"->C), which was the original bug: verbose MCQ
    # answers like "<answer>D. Cabinet desk</answer>" returned 'C'.
    m = re.search(rf'<\|begin_of_box\|>\s*({lr})\s*<\|end_of_box\|>',
                  text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for tag in ('answer', 'Answer'):
        for tm in re.finditer(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL):
            standalone = re.findall(rf'\b({lr})\b', tm.group(1))
            if standalone:
                return standalone[-1]

    # 3. A. / B. (mindcube-style, last occurrence) and `X` (MMSI backtick).
    # Below the tag/box scan so an explicit wrapper wins, but above the
    # natural-language patterns so "A. option-text" outputs still parse.
    hits = list(re.finditer(rf'\b({lr})\.', text))
    if hits:
        return hits[-1].group(1).upper()
    m = re.search(rf'`\s*({lr})\s*`', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4. Natural-language patterns — last match
    for pat in (
        rf'[Mm]y answer is\s*\**\s*({lr})\b',
        rf'[Tt]he answer is\s*\**\s*({lr})\b',
        rf'(?:[Aa]nswer|ANSWER)\s*[:=]\s*\**\s*({lr})\b',
        rf'(?:option|Option|OPTION)\s+({lr})\b',
    ):
        hits = list(re.finditer(pat, text))
        if hits:
            return hits[-1].group(1).upper()

    # 5. (A) / A: — last occurrence
    for pat in (
        rf'\(({lr})\)',
        rf'\b({lr}):',
    ):
        hits = list(re.finditer(pat, text))
        if hits:
            return hits[-1].group(1).upper()

    # 6. Final fallback — last standalone letter anywhere in the string
    hits = list(re.finditer(rf'\b({lr})\b', text))
    if hits:
        return hits[-1].group(1).upper()

    return None


@dataclass
class MMSIBenchSample(BaseBenchmarkSample):
    thought: str


class MMSIBench(BaseBenchmark):
    data_specific_prompt: str = "Answer with the option's letter from the given choices. Enclose the option's letter within \\boxed{}."
    # MMSI-Bench's official instruction (github.com/InternRobotics/MMSI-Bench,
    # also used verbatim by the EASI spatial-reasoning evaluation suite).
    official_data_specific_prompt: str = (
        "Answer with the option's letter from the given choices directly. "
        "Enclose the option's letter within ``."
    )
    valid_question_types = [
        'Attribute (Appr.)',
        'Attribute (Meas.)', 
        'Motion (Cam.)', 
        'Motion (Obj.)', 
        'Positional Relationship (Cam.–Cam.)',
        'Positional Relationship (Cam.–Obj.)',
        'Positional Relationship (Cam.–Reg.)', 
        'Positional Relationship (Obj.–Obj.)',
        'Positional Relationship (Obj.–Reg.)',
        'Positional Relationship (Reg.–Reg.)', 
        'MSR',
    ]

    def __init__(self, data_path: str, question_type: List[str] = None):
        super().__init__()
        
        if question_type is None or 'all' in question_type:
            valid_types = self.valid_question_types
            invalid_types = []
        else:
            valid_types, invalid_types = [], []
            for qt in question_type:
                if qt in self.valid_question_types:
                    valid_types.append(qt)
                else:
                    invalid_types.append(qt)

        if not valid_types:
            raise ValueError(
                f'question_type {question_type} not supported. Expected {self.valid_question_types}.'
            )
        if invalid_types:
            print(
                f'[Warning] partial question_type {invalid_types} not supported. Expected '
                f'{self.valid_question_types}.'
            )
        self.question_type = valid_types
        
        self.data_path = data_path
        self.data, self.image_paths = self.read_data()

    def dump_images(self, data: pd.DataFrame):
        image_dir = os.path.join(self.data_path, 'images')
        os.makedirs(image_dir, exist_ok=True)

        for _, row in data.iterrows():
            id_val = row['id']
            images = row['images']  
            if images is None:
                continue

            for n, img_data in enumerate(images):
                image_path = os.path.join(image_dir, f'{id_val}_{n}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(img_data)

    def read_data(self) -> pd.DataFrame:
        parquet_path = os.path.join(self.data_path, 'MMSI_Bench.parquet')
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f'Data file of MMSI-Bench not exists: {parquet_path}')
        image_dir = os.path.join(self.data_path, 'images')
        if not os.path.exists(image_dir):
            self.dump_images(pd.read_parquet(parquet_path, columns=['id', 'images']))

        # the embedded jpg bytes ('images') dominate the parquet (~GBs in RAM);
        # once dumped to image_dir they are never needed again, so skip the column
        import pyarrow.parquet as pq
        columns = [c for c in pq.ParquetFile(parquet_path).schema_arrow.names if c != 'images']
        data = pd.read_parquet(parquet_path, columns=columns)

        data = data[data['question_type'].isin(self.question_type)]
        print('Evaluating question types:')
        for qt in self.question_type:
            print(qt)
        print(f'Totally {len(data)} samples.')

        image_paths = {}
        for _, row in data.iterrows():
            id_val = row['id']
            image_paths_i = []
            while os.path.exists(f'{image_dir}/{id_val}_{len(image_paths_i)}.jpg'):
                image_paths_i.append(f'{image_dir}/{id_val}_{len(image_paths_i)}.jpg')
            image_paths[id_val] = image_paths_i

        return data, image_paths
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> MMSIBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        
        row = self.data.iloc[index]
        return MMSIBenchSample(
            sample_id=row['id'],
            images=self.image_paths.get(row['id'], []),
            question_type=row['question_type'],
            question=row['question'],
            answer=row['answer'],
            thought=row.get('thought', ''),
        )

    def get_subset(self, question_type: str) -> str:
        if question_type.startswith('Positional'):
            return 'Positional Relationship'
        if question_type.startswith('Motion'):
            return 'Motion'
        if question_type.startswith('Attribute'):
            return 'Attribute'
        return 'MSR'

    def extract_answer(self, prediction: str, question_text: str = '') -> Optional[str]:
        """Extract the option letter (A-D); see `extract_letter`."""
        return extract_letter(prediction, 'ABCD')

    def evaluate(
        self, 
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        if ignore_empty:
            predictions = {sample_id: content for sample_id, content in predictions.items() if content.strip()}

        results = {
            'total_samples': 0,
            'correct_samples': 0,
            'parse_fail_samples': 0,
            'overall_accuracy': 0.0,
            'parse_fail_rate': 0.0,
            'detailed_results': [],
        }
        for subset in ['Positional Relationship', 'Motion', 'Attribute', 'MSR']:
            results[subset] = {
                'total_samples': 0,
                'correct_samples': 0,
                'parse_fail_samples': 0,
                'overall_accuracy': 0.0,
                'parse_fail_rate': 0.0,
                'question_type_accuracy': {},
            }

        for _, row in self.data.iterrows():
            sample_id = row['id']
            ground_truth = str(row['answer']).strip().upper()
            qtype = row['question_type']
            subset = self.get_subset(qtype)

            prediction = predictions.get(sample_id, '')
            if ignore_empty and prediction.strip() == '':
                continue

            if qtype not in results[subset]['question_type_accuracy']:
                results[subset]['question_type_accuracy'][qtype] = {
                    'correct_samples': 0, 'total_samples': 0, 'parse_fail_samples': 0,
                }

            extracted_answer = self.extract_answer(prediction, row['question'])
            is_correct = (extracted_answer == ground_truth) if extracted_answer else False

            results['total_samples'] += 1
            results[subset]['total_samples'] += 1
            results[subset]['question_type_accuracy'][qtype]['total_samples'] += 1
            if is_correct:
                results['correct_samples'] += 1
                results[subset]['correct_samples'] += 1
                results[subset]['question_type_accuracy'][qtype]['correct_samples'] += 1
            if extracted_answer is None:
                results['parse_fail_samples'] += 1
                results[subset]['parse_fail_samples'] += 1
                results[subset]['question_type_accuracy'][qtype]['parse_fail_samples'] += 1

            results['detailed_results'].append({
                'id': sample_id,
                'subset': subset,
                'question_type': qtype,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct
            })

        results['overall_accuracy'] = results['correct_samples'] / results['total_samples']
        results['parse_fail_rate'] = results['parse_fail_samples'] / results['total_samples']
        for subset in ['Positional Relationship', 'Motion', 'Attribute', 'MSR']:
            if results[subset]['total_samples'] == 0:
                del results[subset]
            else:
                results[subset]['overall_accuracy'] = (
                    results[subset]['correct_samples'] / results[subset]['total_samples']
                )
                results[subset]['parse_fail_rate'] = (
                    results[subset]['parse_fail_samples'] / results[subset]['total_samples']
                )
                for qt, s in results[subset]['question_type_accuracy'].items():
                    results[subset]['question_type_accuracy'][qt]['accuracy'] = (
                        s['correct_samples'] / s['total_samples']
                    )
                    results[subset]['question_type_accuracy'][qt]['parse_fail_rate'] = (
                        s['parse_fail_samples'] / s['total_samples']
                    )

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results.csv')
            self.save_results(results, output_file)
        self.pretty_print_results(results, output_dir)
        return results
    
    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 64)
        print('MMSI-Bench Evaluation Results')
        print('=' * 64)
        
        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correrct samples: {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')
        print(f'Parse-fail rate : {results["parse_fail_rate"]:6.2%} '
              f'({results["parse_fail_samples"]}/{results["total_samples"]})')

        print('=' * 64)
        print('Accuracy by Question Type:')
        print('=' * 64)
        for subset in ['Positional Relationship', 'Motion', 'Attribute', 'MSR']:
            if subset not in results:
                continue

            subset_res = results[subset]
            print(
                f'- {subset}: {subset_res["overall_accuracy"]:7.2%} '
                f'({subset_res["correct_samples"]:3d}/{subset_res["total_samples"]:3d}) '
                f'parse_fail {subset_res["parse_fail_rate"]:6.2%} ({subset_res["parse_fail_samples"]:3d})'
            )
            for qt, s in subset_res['question_type_accuracy'].items():
                print(
                    f'    {qt:40} {s["accuracy"]:6.2%} ({s["correct_samples"]:3d}/{s["total_samples"]:3d}) '
                    f'parse_fail {s["parse_fail_rate"]:6.2%} ({s["parse_fail_samples"]:3d})'
                )
            print('=' * 64)

        # Build structured JSON summary
        summary = {
            'total_samples': int(results['total_samples']),
            'correct_samples': int(results['correct_samples']),
            'parse_fail_samples': int(results['parse_fail_samples']),
            'overall_accuracy': round(float(results['overall_accuracy']), 4),
            'parse_fail_rate': round(float(results['parse_fail_rate']), 4),
            'subset': {
                str(s): {
                    'accuracy': round(float(results[s]['overall_accuracy']), 4),
                    'parse_fail_rate': round(float(results[s]['parse_fail_rate']), 4),
                    'correct_samples': int(results[s]['correct_samples']),
                    'parse_fail_samples': int(results[s]['parse_fail_samples']),
                    'total_samples': int(results[s]['total_samples']),
                    'details': {
                        str(q): {
                            'accuracy': round(float(v['accuracy']), 4),
                            'parse_fail_rate': round(float(v['parse_fail_rate']), 4),
                            'correct_samples': int(v['correct_samples']),
                            'parse_fail_samples': int(v['parse_fail_samples']),
                            'total_samples': int(v['total_samples']),
                        }
                        for q, v in results[s]['question_type_accuracy'].items()
                    }
                }
                for s in ['Positional Relationship', 'Motion', 'Attribute', 'MSR']
                if s in results
            }
        }

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
