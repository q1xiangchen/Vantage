import json
import os
import re
import string
from typing import Dict, List, Optional

import pandas as pd

from evals.parser import extract_answer as parse_final_choice
from evals.base import BaseBenchmark, BaseBenchmarkSample


# Scoring from lmms-eval. See licenses/lmms-eval.txt.
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c26b6b2713c3ec155962ff990f956bf78e0efe45/lmms_eval/tasks/blink/utils.py
def _extract_answer_letter(text: str, valid_letters: str = string.ascii_uppercase) -> str:
    """Extract a valid option letter from a response."""
    text = text.strip()
    letters = re.escape(valid_letters.upper())

    # 1) Terse response: the first line is (almost) nothing but the letter.
    first_line = text.split("\n", 1)[0].strip()
    if len(first_line) <= 4:
        match = re.match(rf"^\(?([{letters}])\)?[\.\s:,)]*$", first_line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 2) Explicit statement, e.g. "the correct answer is (B)"; use the last one.
    matches = re.findall(rf"answers?(?:\s+is)?\s*[:\-]?\s*\(?([{letters}])\)?(?![A-Za-z])", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 3) Parenthesized option anywhere, e.g. "... so I pick (C)"; use the last one.
    matches = re.findall(rf"\(([{letters}])\)", text)
    if matches:
        return matches[-1].upper()

    # 4) Leading letter, restricted to the valid choices.
    match = re.match(rf"[\(\s]*([{letters}])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""

def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "blink_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    # extract predicted answer, accepting only this doc's option letters
    valid_letters = string.ascii_uppercase[: len(doc.get("choices") or [])] or string.ascii_uppercase
    pred_letter = _extract_answer_letter(response, valid_letters)
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "sub_task": doc["sub_task"], "is_correct": flag}
    return {key_name: omnispatial_submission}

def blink_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


class BlinkBench(BaseBenchmark):
    # BLINK paper uses the dataset's `prompt` field verbatim as the model input
    # (already contains "Select from the following options.\n(A) left\n(B) right").
    # Keep data_specific_prompt empty so nothing is appended to that wording.
    data_specific_prompt: str = ''
    official_data_specific_prompt: str = ''

    # Only the multi-view reasoning subset is loaded here — kept as a list so
    # more BLINK subsets can be added later without changing the plumbing.
    subset_files = {
        'multiview_reasoning': 'Multi-view_Reasoning/val-00000-of-00001.parquet',
    }
    valid_question_types = list(subset_files)

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
        # multiview_reasoning is prepared into data/blink/multiview_reasoning_val.json
        # (see docs/dataset.md). Read that instead of the parquet at run time to
        # avoid a pyarrow dep in the eval hot path.
        json_path = os.path.join(self.data_path, 'multiview_reasoning_val.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f'BLINK data file not found: {json_path}')
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        # answer strings are '(A)' / '(B)'; strip to a bare letter.
        def clean_answer(s: str) -> str:
            m = re.search(r'\(([A-Z])\)', s)
            return m.group(1).upper() if m else str(s).strip()[0].upper()

        rows = []
        for rec in data_json:
            if 'multiview_reasoning' not in self.question_type:
                continue
            rows.append({
                'id': rec['idx'],
                'question_type': 'multiview_reasoning',
                'prompt': rec['prompt'],
                'choices': list(rec['choices']),
                'answer': clean_answer(rec['answer']),
                'image_paths': rec['image_paths'],
            })
        data = pd.DataFrame(rows)

        print('Evaluating question types:')
        for qt in self.question_type:
            print(f'- {qt}')
        print(f'Totally {len(data)} samples.')

        image_paths = {
            row['id']: [os.path.join(self.data_path, img) for img in row['image_paths']]
            for _, row in data.iterrows()
        }
        return data, image_paths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> BaseBenchmarkSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        row = self.data.iloc[index]
        # Paper-alignment: feed the raw `prompt` field verbatim (already contains
        # the (A) / (B) options). No extra formatting or letter-in-boxed hint.
        return BaseBenchmarkSample(
            sample_id=row['id'],
            images=self.image_paths.get(row['id'], []),
            question_type=row['question_type'],
            question=row['prompt'],
            answer=row['answer'],
        )

    def extract_answer(self, prediction: str, question_text: str = '', choices=None) -> str:
        """Use the shared final-choice parser, restricted to the offered options."""
        if choices is None:
            # Compatibility for callers using the historical two-argument API.
            choices = re.findall(r'\(([A-Z])\)', question_text)
        valid_letters = string.ascii_uppercase[:len(choices)] or string.ascii_uppercase
        return parse_final_choice(prediction, valid_letters)

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
            'overall_accuracy': 0.0,
            'detailed_results': [],
        }
        for subset in self.valid_question_types:
            results[subset] = {
                'total_samples': 0,
                'correct_samples': 0,
                'overall_accuracy': 0.0,
            }

        for _, row in self.data.iterrows():
            sid = row['id']
            qtype = row['question_type']
            gt = row['answer']

            pred = predictions.get(sid, '')
            if ignore_empty and (not pred or str(pred).strip() == ''):
                continue

            scored = blink_process_results({
                'idx': sid,
                'answer': gt,
                'choices': row['choices'],
                'sub_task': qtype,
            }, [self.extract_answer(pred, row['prompt'], row['choices'])])['blink_acc']
            extracted = scored['pred_parsed']
            is_correct = scored['is_correct']

            if is_correct:
                results['correct_samples'] += 1
                results[qtype]['correct_samples'] += 1
            results['total_samples'] += 1
            results[qtype]['total_samples'] += 1

            results['detailed_results'].append({
                'id': sid,
                'question_type': qtype,
                'ground_truth': gt,
                'prediction': pred,
                'extracted_answer': extracted,
                'is_correct': is_correct,
            })

        results['overall_accuracy'] = blink_aggregate_results(results['detailed_results'])
        for subset in self.valid_question_types:
            if results[subset]['total_samples'] == 0:
                del results[subset]
            else:
                results[subset]['overall_accuracy'] = (
                    results[subset]['correct_samples'] / results[subset]['total_samples']
                )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.save_results(results, os.path.join(output_dir, 'results.csv'))
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('BLINK Evaluation Results')
        print('=' * 60)

        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correct samples : {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')

        print('=' * 60)
        print('Accuracy by Question Type:')
        print('=' * 60)
        for subset in self.valid_question_types:
            if subset not in results:
                continue
            subset_res = results[subset]
            print(
                f'- {subset}: {subset_res["overall_accuracy"]:7.2%} '
                f'({subset_res["correct_samples"]:4d}/{subset_res["total_samples"]:4d})'
            )
        print('=' * 60)

        summary = {
            'total_samples': int(results['total_samples']),
            'correct_samples': int(results['correct_samples']),
            'overall_accuracy': round(float(results['overall_accuracy']), 4),
            'subset': {
                str(s): {
                    'accuracy': round(float(results[s]['overall_accuracy']), 4),
                    'correct_samples': int(results[s]['correct_samples']),
                    'total_samples': int(results[s]['total_samples']),
                }
                for s in self.valid_question_types
                if s in results
            },
        }

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
