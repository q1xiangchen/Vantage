from datetime import datetime
import json
import os
from typing import Any, Dict, Optional

from PIL import Image

from workflow.config import get_config


def safe_default(o):
    return f'<non-serializable type: {type(o).__name__}>'


def build_trace_record(
    *,
    mode: str,
    route: str,
    origin_question: str,
    analysis_raw: Optional[str] = None,
    analysis_parsed: Optional[Dict[str, Any]] = None,
    planning_raw: Optional[str] = None,
    planning_parsed: Optional[Dict[str, Any]] = None,
    viewpoint: Optional[Dict[str, Any]] = None,
    synthesized: bool = False,
    synthesis_error: Optional[str] = None,
    analysis_error: Optional[str] = None,
    planning_error: Optional[str] = None,
    answer: Optional[str] = None,
    answer_prompt: Optional[str] = None,
    answer_thinking: Optional[str] = None,
    synthesis_diag: Optional[Dict[str, Any]] = None,
    vqa_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble analysis, planning, synthesis, and answer details for one sample."""
    record: Dict[str, Any] = {
        'mode': mode,
        'route': route,
        'origin_question': origin_question,
        'viewpoint': viewpoint,
        'synthesized': synthesized,
        'synthesis_error': synthesis_error,
        'synthesis_diag': synthesis_diag or None,
        'question_analysis': None,
        'view_planning': None,
        'vqa_answer': {
            'answer': answer,
            'thinking': answer_thinking,
            'answer_prompt': answer_prompt,
        },
    }
    if analysis_raw is not None or analysis_parsed is not None or analysis_error is not None:
        record['question_analysis'] = {'raw': analysis_raw, 'parsed': analysis_parsed}
        if analysis_error is not None:
            record['question_analysis']['error'] = analysis_error
    if planning_raw is not None or planning_parsed is not None or planning_error is not None:
        record['view_planning'] = {'raw': planning_raw, 'parsed': planning_parsed}
        if planning_error is not None:
            record['view_planning']['error'] = planning_error
    if vqa_error is not None:
        record['vqa_answer']['error'] = vqa_error
    return record


class AgentLogger:

    def get_session_dir(self, session_id: str) -> str:
        session_dir = os.path.join(get_config().work_dir, 'sessions', f'session-{session_id}')
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _trace_node(self, session_id: str, log_entry: Dict[str, Any]):
        session_dir = self.get_session_dir(session_id)
        log_entry['timestamp'] = datetime.utcnow().isoformat()
        with open(os.path.join(session_dir, 'trace.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=safe_default) + '\n')

    def log_trace(
        self,
        session_id: str,
        record: Dict[str, Any],
        synth_image: Optional[Image.Image] = None,
    ) -> None:
        """Write a sample trace to its session directory."""
        session_dir = self.get_session_dir(session_id)
        record = dict(record)

        if synth_image is not None:
            viz_dir = os.path.join(session_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            synth_path = os.path.join(viz_dir, 'synth_view.png')
            synth_image.save(synth_path)
            record['synth_image_path'] = os.path.relpath(synth_path, session_dir)

        self._trace_node(session_id, {'event_type': 'vantage', **record})
