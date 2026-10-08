from dataclasses import dataclass
import json
import os
import re
import string
from typing import Dict, List, Optional, Tuple

import pandas as pd

from evals.base import BaseBenchmark, BaseBenchmarkSample


@dataclass
class SpinBenchSample(BaseBenchmarkSample):
    task_type: str
    dataset_type: str
    n_options: int


class SpinBench(BaseBenchmark):
    data_specific_prompt: str = (
        'Answer with the option\'s letter (A, B, C, D). '
        'Enclose the option\'s letter within \\boxed{}.'
    )

    # The 7 official spatial reasoning categories from the SpinBench paper
    # (arXiv 2509.25390, Table 4), rolled up from the 51 fine-grained metadata
    # task_types. Official metrics: raw accuracy + Cohen's kappa, where kappa
    # chance-corrects by each task's option cardinality (also from Table 4).
    # The paper's third metric, pairwise consistency, needs augmentation-pair
    # ids that the released test.jsonl does not carry, so it is not computed.
    valid_question_types = [
        'identity_matching',
        'object_relation_grounding',
        'dynamic_rotation',
        'dynamic_translation',
        'canonical_view_selection',
        'perspective_taking',
        'mental_rotation',
    ]

    @staticmethod
    def classify_task(task_type: str) -> Tuple[str, int]:
        """Fine-grained metadata task_type -> (official category, #options)."""
        tt = task_type
        if 'mental_rotation' in tt:
            return 'mental_rotation', 4
        if 'rotation_classification' in tt:
            return 'dynamic_rotation', 2
        if 'canonical_view_selection' in tt:
            return 'canonical_view_selection', 2 if tt.startswith('face') else 3
        if 'rotation_selection' in tt:
            return 'perspective_taking', 3
        if 'spatial_relation_transformation' in tt:
            return 'perspective_taking', 2
        if 'spatial_relationship_dynamic' in tt:
            return 'dynamic_translation', 2
        if 'spatial_relation_grounding' in tt or 'spatial_relationship_front_behind' in tt:
            return 'object_relation_grounding', 2
        if 'identity' in tt:
            return 'identity_matching', 4 if 'quartet' in tt else 3
        raise ValueError(f'Unrecognized SpinBench task_type: {task_type}')

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

    def read_data(self):
        jsonl_path = os.path.join(self.data_path, 'test.jsonl')
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f'SpinBench test.jsonl not found: {jsonl_path}')

        image_dir = os.path.join(self.data_path, 'images')
        if not os.path.exists(image_dir):
            raise FileNotFoundError(
                f'Image folder of SpinBench not exists: {image_dir} (unzip images.zip)'
            )

        rows = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            # Line number is the sample id: stable across question_type subsets.
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                meta = rec.get('metadata') or {}
                task_type = meta.get('task_type', '')
                category, n_options = self.classify_task(task_type)
                rows.append({
                    'id': idx,
                    'question': rec['problem'],
                    'answer': str(rec['answer']).strip().upper(),
                    'images': rec.get('images') or [],
                    'task_type': task_type,
                    'dataset_type': meta.get('dataset_type') or '',
                    'question_type': category,
                    'n_options': n_options,
                })

        data = pd.DataFrame(rows)
        data = data[data['question_type'].isin(self.question_type)]

        print('Evaluating question types:')
        for qt in self.question_type:
            print(f'- {qt}')
        print(f'Totally {len(data)} samples.')

        image_paths = {
            row['id']: [os.path.join(self.data_path, img) for img in row['images']]
            for _, row in data.iterrows()
        }
        return data, image_paths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> SpinBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        row = self.data.iloc[index]

        # Problems interleave `<image>` placeholders with text; the workflow attaches
        # images separately in order, so relabel placeholders to match that order.
        counter = iter(range(1, len(row['images']) + 1))
        question = re.sub(r'<image>', lambda m: f'[Image {next(counter)}]', row['question'])

        return SpinBenchSample(
            sample_id=int(row['id']),
            images=self.image_paths.get(row['id'], []),
            question_type=row['question_type'],
            task_type=row['task_type'],
            dataset_type=row['dataset_type'],
            n_options=int(row['n_options']),
            question=question,
            answer=row['answer'],
        )

    def extract_answer(self, prediction: str) -> Optional[str]:
        if prediction is None:
            return None
        prediction = str(prediction).strip()

        match = re.search(r'\\boxed{\s*([A-D])\s*}', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r'<\|begin_of_box\|>\s*([A-D])\s*<\|end_of_box\|>', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Official evaluator's primary pattern (eval/vl_benchmark_evaluator.py).
        match = re.search(r'<answer>\s*([A-D])\s*</answer>', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        patterns = [
            r'\b([A-D])\.',
            r'\(([A-D])\)',
            r'\b([A-D]):',
        ]
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        cleaned = prediction.translate(
            str.maketrans('', '', string.punctuation)
        ).replace(' ', '')
        if len(cleaned) == 1 and cleaned.upper() in ['A', 'B', 'C', 'D']:
            return cleaned.upper()

        return None

    @staticmethod
    def _kappa(correct: float, total: float, chance_sum: float) -> float:
        """Cohen's kappa with per-sample chance 1/#options (chance_sum = sum of them)."""
        if total == 0:
            return 0.0
        p_o = correct / total
        p_e = chance_sum / total
        return (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0

    def evaluate(
        self,
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        if ignore_empty:
            predictions = {k: v for k, v in predictions.items() if str(v).strip()}

        results = {
            'total_samples': 0,
            'correct_samples': 0,
            'parse_fail_samples': 0,
            'chance_sum': 0.0,
            'overall_accuracy': 0.0,
            'parse_fail_rate': 0.0,
            'overall_kappa': 0.0,
            'detailed_results': [],
        }
        for subset in self.valid_question_types:
            results[subset] = {
                'total_samples': 0,
                'correct_samples': 0,
                'parse_fail_samples': 0,
                'chance_sum': 0.0,
                'overall_accuracy': 0.0,
                'parse_fail_rate': 0.0,
                'overall_kappa': 0.0,
                'question_type_accuracy': {},
            }

        for _, row in self.data.iterrows():
            sid = int(row['id'])
            gt = row['answer']
            qtype = row['task_type']
            subset = row['question_type']
            chance = 1.0 / row['n_options']

            pred = predictions.get(sid, '')
            if ignore_empty and (not pred or str(pred).strip() == ''):
                continue

            if qtype not in results[subset]['question_type_accuracy']:
                results[subset]['question_type_accuracy'][qtype] = {
                    'correct_samples': 0, 'total_samples': 0, 'parse_fail_samples': 0,
                    'chance_sum': 0.0,
                }
            qt_stats = results[subset]['question_type_accuracy'][qtype]

            extracted = self.extract_answer(pred)
            is_correct = extracted == gt
            if is_correct:
                results['correct_samples'] += 1
                results[subset]['correct_samples'] += 1
                qt_stats['correct_samples'] += 1
            if extracted is None:
                results['parse_fail_samples'] += 1
                results[subset]['parse_fail_samples'] += 1
                qt_stats['parse_fail_samples'] += 1
            results['total_samples'] += 1
            results[subset]['total_samples'] += 1
            qt_stats['total_samples'] += 1
            results['chance_sum'] += chance
            results[subset]['chance_sum'] += chance
            qt_stats['chance_sum'] += chance

            results['detailed_results'].append({
                'id': sid,
                'question_type': qtype,
                'category': subset,
                'n_options': int(row['n_options']),
                'ground_truth': gt,
                'prediction': pred,
                'extracted_answer': extracted,
                'is_correct': is_correct,
            })

        results['overall_accuracy'] = (
            results['correct_samples'] / results['total_samples'] if results['total_samples'] else 0
        )
        results['parse_fail_rate'] = (
            results['parse_fail_samples'] / results['total_samples'] if results['total_samples'] else 0
        )
        results['overall_kappa'] = self._kappa(
            results['correct_samples'], results['total_samples'], results['chance_sum']
        )
        for subset in self.valid_question_types:
            if results[subset]['total_samples'] == 0:
                del results[subset]
            else:
                results[subset]['overall_accuracy'] = (
                    results[subset]['correct_samples'] / results[subset]['total_samples']
                )
                results[subset]['parse_fail_rate'] = (
                    results[subset]['parse_fail_samples'] / results[subset]['total_samples']
                )
                results[subset]['overall_kappa'] = self._kappa(
                    results[subset]['correct_samples'],
                    results[subset]['total_samples'],
                    results[subset]['chance_sum'],
                )
                for qt, s in results[subset]['question_type_accuracy'].items():
                    s['accuracy'] = s['correct_samples'] / s['total_samples']
                    s['parse_fail_rate'] = s['parse_fail_samples'] / s['total_samples']
                    s['kappa'] = self._kappa(
                        s['correct_samples'], s['total_samples'], s['chance_sum']
                    )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.save_results(results, os.path.join(output_dir, 'results.csv'))
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('SpinBench Evaluation Results')
        print('=' * 60)

        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correct samples : {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')
        print(f'Overall kappa   : {results["overall_kappa"]:6.3f}')
        print(f'Parse-fail rate : {results["parse_fail_rate"]:6.2%} '
              f'({results["parse_fail_samples"]}/{results["total_samples"]})')

        print('=' * 60)
        print('Accuracy / Cohen\'s kappa by Category:')
        print('=' * 60)
        for subset in self.valid_question_types:
            if subset not in results:
                continue

            subset_res = results[subset]
            print(
                f'- {subset}: acc {subset_res["overall_accuracy"]:7.2%} '
                f'kappa {subset_res["overall_kappa"]:6.3f} '
                f'({subset_res["correct_samples"]:4d}/{subset_res["total_samples"]:4d}) '
                f'parse_fail {subset_res["parse_fail_rate"]:6.2%} ({subset_res["parse_fail_samples"]:3d})'
            )
            for qt, s in subset_res['question_type_accuracy'].items():
                print(
                    f'    {qt:55} acc {s["accuracy"]:6.2%} kappa {s["kappa"]:6.3f} '
                    f'({s["correct_samples"]:3d}/{s["total_samples"]:3d}) '
                    f'parse_fail {s["parse_fail_rate"]:6.2%} ({s["parse_fail_samples"]:3d})'
                )
            print('=' * 60)

        summary = {
            'total_samples': int(results['total_samples']),
            'correct_samples': int(results['correct_samples']),
            'parse_fail_samples': int(results['parse_fail_samples']),
            'overall_accuracy': round(float(results['overall_accuracy']), 4),
            'parse_fail_rate': round(float(results['parse_fail_rate']), 4),
            'overall_kappa': round(float(results['overall_kappa']), 4),
            'subset': {
                str(s): {
                    'accuracy': round(float(results[s]['overall_accuracy']), 4),
                    'parse_fail_rate': round(float(results[s]['parse_fail_rate']), 4),
                    'kappa': round(float(results[s]['overall_kappa']), 4),
                    'correct_samples': int(results[s]['correct_samples']),
                    'parse_fail_samples': int(results[s]['parse_fail_samples']),
                    'total_samples': int(results[s]['total_samples']),
                    'details': {
                        str(q): {
                            'accuracy': round(float(v['accuracy']), 4),
                            'parse_fail_rate': round(float(v['parse_fail_rate']), 4),
                            'kappa': round(float(v['kappa']), 4),
                            'correct_samples': int(v['correct_samples']),
                            'parse_fail_samples': int(v['parse_fail_samples']),
                            'total_samples': int(v['total_samples']),
                        }
                        for q, v in results[s]['question_type_accuracy'].items()
                    }
                }
                for s in self.valid_question_types
                if s in results
            },
        }

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
