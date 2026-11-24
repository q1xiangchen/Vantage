"""Evaluate SPAR-Bench responses with the benchmark’s scoring and aggregation rules."""
from dataclasses import dataclass
import json
import logging
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evals.parser import extract_answer as parse_final_choice
from evals.base import BaseBenchmark, BaseBenchmarkSample


# Scoring adapted from lmms-eval. See licenses/lmms-eval.txt.
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c26b6b2713c3ec155962ff990f956bf78e0efe45/lmms_eval/tasks/sparbench/utils.py
eval_logger = logging.getLogger("lmms-eval")

MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]

SPECIAL_QUESTION_TYPES = [
    "view_change_infer",
]

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

Low = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]

Middle = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

High = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

def process_na(pred, task):
    numbers = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)

    # Convert the matched numbers to float or int
    extracted_numbers = [float(num) if "." in num else int(num) for num in numbers]
    if task in [
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
    ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]

def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}

def compute_vci_metric(pred, answer):
    _acion_list = ["move_right", "move_left", "move_forward", "move_backward", "move_up", "move_down", "rotate_right", "rotate_left", "rotate_up", "rotate_down"]
    action_order = ["move_right_left", "move_up_down", "move_forward_backward", "rotate_right_left", "rotate_up_down"]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []

    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))

    mra_list = []
    for gt, answer in zip(gt_list, answer_list):
        mra = mean_relative_accuracy(gt, answer, start=0.5, end=0.95, interval=0.05)
        mra_list.append(mra)

    return np.mean(mra_list)

def abs_dist_norm(pred, target):
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred

def _compute_all_subscores(results) -> dict:
    """Compute all sub-category scores from raw results. Shared logic for all aggregation functions."""
    df = pd.DataFrame(results)
    output = {}

    for question_type, question_type_indexes in df.groupby("task").groups.items():
        per_question_type = df.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in SPECIAL_QUESTION_TYPES:
            if question_type == "view_change_infer":
                output[f"{question_type}_vci_metric"] = per_question_type["vci_metric"].mean()

    output["overall"] = sum([_ for _ in output.values()]) / len(output) if output else 0.0

    low_list = []
    middle_list = []
    high_list = []
    for task_key in list(output.keys()):
        if task_key == "overall":
            continue
        task_name = "_".join(task_key.split("_")[:-1])
        if task_name in Low:
            low_list.append(output[task_key])
        elif task_name in Middle:
            middle_list.append(output[task_key])
        elif task_name in High:
            high_list.append(output[task_key])

    output["low"] = np.mean(low_list) if low_list else 0.0
    output["middle"] = np.mean(middle_list) if middle_list else 0.0
    output["high"] = np.mean(high_list) if high_list else 0.0

    return output

def sparbench_aggregate_results(results):
    output = _compute_all_subscores(results)
    eval_logger.info(f"Evaluation results: {output}")
    return output["overall"] * 100.0

def _score_sample(task, kind, prediction, extracted, target):
    if kind == 'select':
        score = float((extracted or '').lower() == target.strip().lower())
        return {'task': task, 'accuracy': score}
    if kind == 'fill':
        try:
            score = mean_relative_accuracy(to_float(process_na(prediction, task)),
                                           to_float(target), .5, .95, .05)
        except (ValueError, TypeError):
            score = 0.0
        return {'task': task, 'MRA:.5:.95:.05': score}
    try:
        score = compute_vci_metric(prediction, target)
    except Exception:
        score = 0.0
    return {'task': task, 'vci_metric': score}


SELECT_TASKS = [
    'camera_motion_infer',
    'distance_infer_center_oo_mv',
    'obj_spatial_relation_oc_mv',
    'obj_spatial_relation_oo_mv',
    'position_matching',
    'spatial_imagination_oc_mv',
    'spatial_imagination_oo_mv',
]
NA_TASKS = [
    'depth_prediction_oc_mv',
    'depth_prediction_oo_mv',
    'distance_prediction_oc_mv',
    'distance_prediction_oo_mv',
]
VCI_TASKS = ['view_change_infer']
FILL_TASKS = NA_TASKS + VCI_TASKS
MV_TASKS = SELECT_TASKS + FILL_TASKS

# Official difficulty groups, restricted to the multi-view tasks.
LOW = NA_TASKS
MIDDLE = ['view_change_infer', 'position_matching', 'camera_motion_infer']
HIGH = ['obj_spatial_relation_oc_mv', 'obj_spatial_relation_oo_mv',
        'spatial_imagination_oc_mv', 'spatial_imagination_oo_mv',
        'distance_infer_center_oo_mv']

# --- official prompt pieces (sparbench.yaml default kwargs + doc_to_text) -----
PRE_PROMPT = ''
NA_POST_PROMPT = 'Please answer the question using a single word or phrase.'
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
BBOX_NOTE = ('The values represent the bounding box coordinates normalized to a '
             '0-1000 scale, with the top-left corner as the origin of the image.')

@dataclass
class SPARBenchSample(BaseBenchmarkSample):
    kind: str          # 'select' | 'fill' | 'vci'
    format_type: str   # dataset format_type: 'select' | 'fill'
    is_numerical: bool


def task_kind(task: str) -> str:
    if task in SELECT_TASKS:
        return 'select'
    if task in NA_TASKS:
        return 'fill'
    if task in VCI_TASKS:
        return 'vci'
    raise ValueError(f'unknown SPAR task {task!r}')


class SPARBench(BaseBenchmark):
    # The answer rule is inlined per question (see module docstring).
    data_specific_prompt: str = ''
    official_data_specific_prompt: str = ''

    valid_question_types = list(MV_TASKS)
    letters = 'ABCD'

    def __init__(self, data_path: str, question_type: List[str] = None):
        super().__init__()
        if question_type is None or 'all' in question_type:
            valid_types, invalid_types = list(self.valid_question_types), []
        else:
            valid_types, invalid_types = [], []
            for qt in question_type:
                (valid_types if qt in self.valid_question_types else invalid_types).append(qt)
        if not valid_types:
            raise ValueError(
                f'question_type {question_type} not supported. Expected {self.valid_question_types}.')
        if invalid_types:
            print(f'[Warning] partial question_type {invalid_types} not supported. Expected '
                  f'{self.valid_question_types}.')
        self.question_type = valid_types
        self.data_path = data_path
        self.data, self.image_paths = self.read_data()

    # ----- data -----------------------------------------------------------------
    def read_data(self):
        jsonl_path = os.path.join(self.data_path, 'sparbench_test.jsonl')
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f'SPAR-Bench data file not found: {jsonl_path}')
        rows, image_paths = [], {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec['img_type'] != 'multi_view' or rec['task'] not in self.question_type:
                    continue
                sid = f'spar_{int(rec["id"]):05d}'
                rows.append({
                    'id': sid,
                    'question_type': rec['task'],
                    'format_type': rec['format_type'],
                    'source': rec['source'],
                    'question': rec['question'],
                    'answer': str(rec['answer']),
                })
                image_paths[sid] = [os.path.join(self.data_path, p) for p in rec['images']]
        data = pd.DataFrame(rows)
        print('Evaluating question types:')
        for qt in self.question_type:
            print(f'- {qt}')
        print(f'Totally {len(data)} samples.')
        return data, image_paths

    def apply_prompt_style(self, style: str) -> None:
        # The official doc_to_text layout is used for every style.
        return

    def build_question(self, row) -> str:
        """Format a SPAR-Bench question with its task-specific answer instruction."""
        q, task = row['question'], row['question_type']
        kind = task_kind(task)
        if kind == 'fill':
            post = NA_POST_PROMPT
            text = PRE_PROMPT + '\n' + q + '\n' + post
        elif kind == 'select':
            note = BBOX_NOTE if task in ('position_matching', 'camera_motion_infer') else ''
            post = MCA_POST_PROMPT
            text = PRE_PROMPT + '\n' + q + '\n' + note + '\n' + post
        else:
            text = PRE_PROMPT + '\n' + q + '\n' + '' + '\n' + ''
        return text.strip()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> SPARBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        row = self.data.iloc[index]
        kind = task_kind(row['question_type'])
        return SPARBenchSample(
            sample_id=row['id'],
            images=self.image_paths.get(row['id'], []),
            question_type=row['question_type'],
            question=self.build_question(row),
            answer=row['answer'],
            kind=kind,
            format_type=row['format_type'],
            is_numerical=kind != 'select',
        )

    # ----- scoring --------------------------------------------------------------
    def extract_answer(self, prediction: str, kind: str):
        """Diagnostics using upstream functions on the original response."""
        if kind == 'select':
            return parse_final_choice(prediction, 'ABCD') or None
        if kind == 'fill':
            return process_na(prediction, 'depth_prediction_oc_mv')
        try:
            return parse_instruction(prediction)
        except Exception:
            return None

    @staticmethod
    def _aggregate(rows: List[Dict]) -> Dict:
        subset = {}
        for qt in MV_TASKS:
            rs = [r for r in rows if r['question_type'] == qt]
            if not rs:
                continue
            n = len(rs)
            pf = sum(not r['parsed'] for r in rs)
            kind = task_kind(qt)
            subset[qt] = {
                'accuracy': sum(r['score'] for r in rs) / n,
                'metric': {'select': 'accuracy', 'fill': 'MRA:.5:.95:.05',
                           'vci': 'vci_metric (official)'}[kind],
                'format_type': 'select' if kind == 'select' else 'fill',
                'total_samples': n,
                'correct_samples': sum(bool(r['is_correct']) for r in rs),
                'parse_fail_samples': pf,
                'parse_fail_rate': pf / n,
            }

        def _mean(tasks):
            vals = [subset[t]['accuracy'] for t in tasks if t in subset]
            return float(np.mean(vals)) if vals else None

        total = sum(v['total_samples'] for v in subset.values())
        pf_total = sum(v['parse_fail_samples'] for v in subset.values())
        return {
            'total_samples': total,
            'correct_samples': sum(v['correct_samples'] for v in subset.values()),
            'parse_fail_samples': pf_total,
            'overall_accuracy': _mean(MV_TASKS) or 0.,
            'parse_fail_rate': pf_total / total if total else 0.,
            'format': {'select': _mean(SELECT_TASKS), 'fill': _mean(FILL_TASKS)},
            'level': {'Low': _mean(LOW), 'Middle': _mean(MIDDLE), 'High': _mean(HIGH)},
            'pooled_accuracy': (sum(r['score'] for r in rows) / len(rows)) if rows else 0.,
            'subset': subset,
        }

    def evaluate(
        self,
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        if ignore_empty:
            predictions = {k: v for k, v in predictions.items() if str(v).strip()}
        upstream_scores = []
        detailed = []
        for _, row in self.data.iterrows():
            sid, qtype, gt = row['id'], row['question_type'], row['answer']
            pred = predictions.get(sid, '')
            if ignore_empty and (not pred or str(pred).strip() == ''):
                continue
            kind = task_kind(qtype)
            extracted = self.extract_answer(pred, kind)
            scored = _score_sample(qtype, kind, pred, extracted, gt)
            upstream_scores.append(scored)
            metric = {'select': 'accuracy', 'fill': 'MRA:.5:.95:.05', 'vci': 'vci_metric'}[kind]
            score = float(scored[metric])
            parsed = extracted is not None and (kind != 'fill' or extracted != -1)
            detailed.append({
                'id': sid, 'question_type': qtype, 'format_type': row['format_type'],
                'ground_truth': gt, 'prediction': pred,
                'extracted_answer': (json.dumps(extracted) if isinstance(extracted, dict)
                                     else extracted),
                'score': score, 'parsed': parsed,
                # select: exact match; fill / vci: score >= 0.5.
                'is_correct': bool(score == 1.) if kind == 'select' else bool(score >= .5),
            })
        results = self._aggregate(detailed)
        percent = float(sparbench_aggregate_results(upstream_scores)) if upstream_scores else 0.0
        results['overall_accuracy'] = percent / 100.0
        results['upstream_score_percent'] = percent
        results['detailed_results'] = detailed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.save_results(results, os.path.join(output_dir, 'results.csv'))
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        def _f(x):
            return f'{x:6.2%}' if x is not None else '   n/a'
        print('\n' + '=' * 60)
        print('SPAR-Bench (multi-view) Evaluation Results')
        print('=' * 60)
        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Overall (mean of {len(results["subset"])} tasks): '
              f'{_f(results["overall_accuracy"])}')
        print(f'select / fill   : {_f(results["format"]["select"])} / {_f(results["format"]["fill"])}')
        print('Low / Mid / High: ' + ' / '.join(_f(results['level'][k]) for k in ('Low', 'Middle', 'High')))
        print(f'Parse-fail rate : {results["parse_fail_rate"]:6.2%} '
              f'({results["parse_fail_samples"]}/{results["total_samples"]})')
        print('=' * 60)
        for c, v in results['subset'].items():
            print(f'- {c}: {v["accuracy"]:7.2%} [{v["metric"]}] (n={v["total_samples"]:4d}) '
                  f'parse_fail {v["parse_fail_rate"]:6.2%} ({v["parse_fail_samples"]:3d})')
        print('=' * 60)

        def _r(x):
            return round(float(x), 4) if x is not None else None
        summary = {
            'upstream_score_percent': results.get('upstream_score_percent'),
            'total_samples': int(results['total_samples']),
            'correct_samples': int(results['correct_samples']),
            'parse_fail_samples': int(results['parse_fail_samples']),
            'overall_accuracy': _r(results['overall_accuracy']),
            'parse_fail_rate': _r(results['parse_fail_rate']),
            'format': {k: _r(v) for k, v in results['format'].items()},
            'level': {k: _r(v) for k, v in results['level'].items()},
            'pooled_accuracy': _r(results['pooled_accuracy']),
            'overall_definition': ('SPAR-Bench: unweighted mean of selected per-task scores '
                                   '(select = accuracy, NA fill = MRA, view_change_infer = '
                                   'official vci_metric), multi-view tasks only; format = '
                                   'mean over the select / fill tasks'),
            'subset': {c: {
                'accuracy': _r(v['accuracy']), 'metric': v['metric'],
                'format_type': v['format_type'],
                'parse_fail_rate': _r(v['parse_fail_rate']),
                'correct_samples': int(v['correct_samples']),
                'parse_fail_samples': int(v['parse_fail_samples']),
                'total_samples': int(v['total_samples']),
            } for c, v in results['subset'].items()},
        }
        if output_dir is not None:
            with open(os.path.join(output_dir, 'results_summary.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
