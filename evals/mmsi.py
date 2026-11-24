from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from evals.parser import extract_answer as parse_final_choice
from evals.base import BaseBenchmark, BaseBenchmarkSample


# Scoring adapted from lmms-eval. See licenses/lmms-eval.txt.
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c26b6b2713c3ec155962ff990f956bf78e0efe45/lmms_eval/tasks/mmsi_bench/utils.py
eval_logger = logging.getLogger("lmms-eval")

CATEGORY_ALIASES = {
    "Positional Relationship (Obj.-Obj.)": "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Cam.-Obj.)": "Positional Relationship (Cam.–Obj.)",
    "Positional Relationship (Cam.-Cam.)": "Positional Relationship (Cam.–Cam.)",
    "Positional Relationship (Obj.-Reg.)": "Positional Relationship (Obj.–Reg.)",
    "Positional Relationship (Cam.-Reg.)": "Positional Relationship (Cam.–Reg.)",
    "Positional Relationship (Reg.-Reg.)": "Positional Relationship (Reg.–Reg.)",
}

def extract_single_choice_with_word_boundary(pred, gt):
    if pred is None:
        return None

    pred = str(pred)

    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r"`([^`]*)`"
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r"\b[A-F]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred, flags=re.IGNORECASE)
    if match:
        pred = match.group().upper()
    else:
        return None

    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception:
        return 0.0
    return 0.0

def msr_process_results(doc, results):
    """Score one MMSI-Bench prediction."""
    pred = results[0]
    gt = doc["answer"]

    score = extract_single_choice_with_word_boundary(pred, gt)
    category = CATEGORY_ALIASES.get(doc["question_type"], doc["question_type"])
    l2_category = category
    if score is None:
        return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "cannot find answer"}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "cannot find answer"}}
    return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score}}

def msr_aggregate_results(results):
    """Compute mean MMSI-Bench accuracy."""
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score


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
        """Extract the final choice with the shared parser."""
        return parse_final_choice(prediction, 'ABCDEF') or None

    def evaluate(
        self, 
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        if ignore_empty:
            predictions = {sample_id: content for sample_id, content in predictions.items() if content.strip()}

        upstream_scores = []
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
            scored = msr_process_results(
                {'id': sample_id, 'answer': row['answer'], 'question_type': qtype}, [extracted_answer or ''])['average']
            upstream_scores.append(scored)
            is_correct = bool(scored['score'])

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

        results['overall_accuracy'] = msr_aggregate_results(upstream_scores)
        results['parse_fail_rate'] = results['parse_fail_samples'] / results['total_samples'] if results['total_samples'] else 0.0
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
