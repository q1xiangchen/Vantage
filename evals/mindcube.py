from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from evals.parser import extract_answer as parse_final_choice
from evals.base import BaseBenchmark, BaseBenchmarkSample


# Scoring adapted from lmms-eval. See licenses/lmms-eval.txt.
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c26b6b2713c3ec155962ff990f956bf78e0efe45/lmms_eval/tasks/mindcube/utils.py
def extract_answer(text: str) -> Optional[str]:
    """Extract the answer choice from a model response."""
    if not text:
        return None

    # First, try to match simple answer format: A., B., C., D., E. with highest priority
    simple_pattern_matches = list(re.finditer(r"([A-E])\.", text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)

    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r"<Answer>(.*?)(?:<|$)", text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [r"[Mm]y answer is ([A-E])", r"[Mm]y answer is ([A-E])\.", r"[Tt]he answer is ([A-E])", r"(?:Answer: )?([A-E])\.", r"\b([A-E])\b"]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)

    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)",  # Answer at line beginning
        r"[\*\"]([A-E])[\*\"]",  # Answer in quotes or asterisks
        r"\bAnswer:?\s*([A-E])\b",  # Answer following "Answer:"
        r"[Mm]y answer is ([A-E])",  # Added pattern for "My answer is X"
        r"[Mm]y answer is ([A-E])\.",  # Added pattern for "My answer is X."
        r"answer is ([A-E])",  # Added pattern for phrases like "The answer is X"
    ]

    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)

    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split("\n")
    line_matches = []

    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))

    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]

    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r"\b([A-E])\b", line)
        if match:
            return match.group(1)

    return None

def mindcube_process_results(doc, results):
    """Score one MindCube prediction."""
    # extract grounded answer
    grounded_output = doc["gt_answer"]

    # extract predicted answer
    pred = results[0].strip()
    pred_answer = extract_answer(pred)

    score = 1.0 if pred_answer == grounded_output else 0.0

    # get type
    row_id = doc["id"]
    first = row_id.split("_", 1)[0]
    question_type = None
    if first == "among":
        question_type = "among"
    elif first == "rotation":
        question_type = "rotation"
    elif first in ("around", "aroundnew"):
        question_type = "around"
    else:
        question_type = "other"

    return {
        "overall_accuracy": {"score": score},
        "around_accuracy": {
            "score": score,
            "type": question_type,
        },
        "among_accuracy": {
            "score": score,
            "type": question_type,
        },
        "rotation_accuracy": {
            "score": score,
            "type": question_type,
        },
    }

def mindcube_aggregate_results(results):
    """Compute mean MindCube accuracy."""
    # --- Compute the total score across all results ---
    total_score = 0.0
    for res in results:
        total_score += res["score"]

    # --- Compute average score safely ---
    avg_score = total_score / len(results) if results else 0.0
    return avg_score


@dataclass
class MindCubeBenchSample(BaseBenchmarkSample):
    category: List[str]
    sample_type: str


# --- MindCube official prompts (github.com/mll-lab-nu/MindCube) --------------
# Verbatim from src/prompt_generation/templates.py. The official prompt layout is
#   <task/rules block>\n[Answer Instruction]...\n[Question]\n<question>
# i.e. instruction BEFORE question — apply_prompt_style therefore rewrites the
# sample questions instead of the appended answer-format suffix.
_OFFICIAL_RAW_QA_BLOCK = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
"""


class MindCubeBench(BaseBenchmark):
    
    data_specific_prompt: str = (
        "Based on these images, answer the question based on this rule: You only need to provide\n"
        "*ONE* correct answer selecting from the options listed below. For example, if you think\n"
        "the correct answer is 'A. above' from ' A. above B. under C. front D. behind.', your\n"
        "response should only be 'A. above'."
    )
    
    valid_question_types = [
        'among',
        'around',
        'rotation'
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
        self.data, self.image_base_dir = self.read_data()

    def read_data(self):
        jsonl_files = ['MindCube_tinybench.jsonl']
        
        search_dirs = [
            self.data_path,
            os.path.join(self.data_path, 'raw'),
        ]
        
        jsonl_path = None
        for search_dir in search_dirs:
            for filename in jsonl_files:
                path = os.path.join(search_dir, filename)
                if os.path.exists(path):
                    jsonl_path = path
                    break
            if jsonl_path:
                break
        
        if jsonl_path is None:
            raise FileNotFoundError(
                f'MindCube data file not found in: {self.data_path}\n'
                f'Searched in: {search_dirs}\n'
                f'Expected one of: {jsonl_files}'
            )
        
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
        
        data = pd.DataFrame(data_list)
        data['scene_type'] = data['id'].apply(self._get_scene_type_static)
        data = data[data['scene_type'].isin(self.question_type)]
        
        print('Evaluating question types:')
        for qt in self.question_type:
            print(f'  - {qt}')
        print(f'Totally {len(data)} samples.')
        
        image_base_dir = os.path.join(self.data_path, 'images')
        if not os.path.exists(image_base_dir):
            alt_paths = [
                os.path.join(self.data_path, 'other_all_image'),
                os.path.join(os.path.dirname(self.data_path), 'other_all_image'),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    image_base_dir = os.path.dirname(alt_path)
                    break
        
        return data, image_base_dir

    def __len__(self) -> int:
        return len(self.data)

    def apply_prompt_style(self, style: str) -> None:
        """Apply the selected benchmark answer format."""
        if style == 'default':
            return
        self.data['question'] = self.data['question'].apply(
            lambda q: f'{_OFFICIAL_RAW_QA_BLOCK}\n[Question]\n{q}')
        self.data_specific_prompt = ''

    def __getitem__(self, index: int) -> MindCubeBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        
        row = self.data.iloc[index]
        
        images = row['images']
        if isinstance(images, str):
            images = json.loads(images)
        
        image_paths = []
        for img in images:
            if not os.path.isabs(img):
                img_path = os.path.join(self.image_base_dir, img)
            else:
                img_path = img
            image_paths.append(img_path)
        
        scene_type = self._get_scene_type(row['id'])
        
        return MindCubeBenchSample(
            sample_id=row['id'],
            question=row['question'],
            images=image_paths,
            answer=row['gt_answer'],
            category=row.get('category', []),
            sample_type=row.get('type', ''),
            question_type=scene_type 
        )

    @staticmethod
    def _get_scene_type_static(sample_id: str) -> str:
        sample_id_lower = str(sample_id).lower()
        
        if 'around' in sample_id_lower:
            return 'around'
        elif 'rotation' in sample_id_lower:
            return 'rotation'
        elif 'translation' in sample_id_lower:
            return 'translation'
        elif 'among' in sample_id_lower:
            return 'among'
        else:
            return 'other'
    
    def _get_scene_type(self, sample_id: str) -> str:
        return self._get_scene_type_static(sample_id)

    def extract_answer(self, prediction: str, question_text: str = '') -> Optional[str]:
        """Extract the final choice with the shared parser."""
        return parse_final_choice(prediction, "ABCDE") or None

    def evaluate(
        self, 
        predictions: Dict[str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        """Score predictions against ground truth."""
        # 
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
            'question_type_accuracy': {},
        }

        for _, row in self.data.iterrows():
            sample_id = row['id']
            ground_truth = str(row['gt_answer']).strip().upper()
            qtype = self._get_scene_type(sample_id)

            prediction = predictions.get(sample_id, '')
            if ignore_empty and prediction.strip() == '':
                continue
            if qtype == 'translation':
                continue

            if qtype not in results['question_type_accuracy']:
                results['question_type_accuracy'][qtype] = {
                    'correct_samples': 0, 'total_samples': 0, 'parse_fail_samples': 0,
                }

            extracted_answer = self.extract_answer(prediction, row['question'])
            scored = mindcube_process_results(
                {'id': sample_id, 'gt_answer': row['gt_answer']}, [extracted_answer or ''])
            upstream_scores.append(scored['overall_accuracy'])
            is_correct = bool(scored['overall_accuracy']['score'])
            if is_correct:
                results['correct_samples'] += 1
                results['question_type_accuracy'][qtype]['correct_samples'] += 1
            if extracted_answer is None:
                results['parse_fail_samples'] += 1
                results['question_type_accuracy'][qtype]['parse_fail_samples'] += 1
            results['total_samples'] += 1
            results['question_type_accuracy'][qtype]['total_samples'] += 1

            results['detailed_results'].append({
                'id': sample_id,
                'question_type': qtype,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct
            })
        
        results['overall_accuracy'] = mindcube_aggregate_results(upstream_scores)
        results['parse_fail_rate'] = results['parse_fail_samples'] / results['total_samples'] if results['total_samples'] else 0.0
        for qt, s in results['question_type_accuracy'].items():
            if s['total_samples'] == 0:
                del results['question_type_accuracy'][qt]
            else:
                results['question_type_accuracy'][qt]['accuracy'] = (
                    s['correct_samples'] / s['total_samples']
                )
                results['question_type_accuracy'][qt]['parse_fail_rate'] = (
                    s['parse_fail_samples'] / s['total_samples']
                )
        
        # Save and print results
        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results.csv')
            self.save_results(results, output_file)
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('MindCube Evaluation Results')
        print('=' * 60)
        
        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correct samples : {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')
        print(f'Parse-fail rate : {results["parse_fail_rate"]:6.2%} '
              f'({results["parse_fail_samples"]}/{results["total_samples"]})')

        print('=' * 60)
        print('Accuracy by Question Type:')
        print('=' * 60)
        for qt, s in results['question_type_accuracy'].items():
            print(
                f'{qt:20} {s["accuracy"]:6.2%} ({s["correct_samples"]:3d}/{s["total_samples"]:3d}) '
                f'parse_fail {s["parse_fail_rate"]:6.2%} ({s["parse_fail_samples"]:3d})'
            )
        print('=' * 60)
        
        # Save JSON summary
        if output_dir is not None:
            summary = {
                'total_samples': int(results['total_samples']),
                'correct_samples': int(results['correct_samples']),
                'parse_fail_samples': int(results['parse_fail_samples']),
                'overall_accuracy': round(float(results['overall_accuracy']), 4),
                'parse_fail_rate': round(float(results['parse_fail_rate']), 4),
                'question_type_accuracy': {
                    str(qt): {
                        'accuracy': round(float(s['accuracy']), 4),
                        'parse_fail_rate': round(float(s['parse_fail_rate']), 4),
                        'correct_samples': int(s['correct_samples']),
                        'parse_fail_samples': int(s['parse_fail_samples']),
                        'total_samples': int(s['total_samples']),
                    }
                    for qt, s in results['question_type_accuracy'].items()
                }
            }
            
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f'Summary saved to: {output_file}')
