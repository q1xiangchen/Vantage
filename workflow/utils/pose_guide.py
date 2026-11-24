"""Session-local pose guide: never recompute facts between planning stages."""
from workflow.config import get_config


def forward_key(session_id):
    return f'sample:{session_id}'


async def fetch_pose_facts(synthesizer, images, key=''):
    if synthesizer is None:
        return {'error': 'view synthesizer unavailable'}
    if len(images) < 2:
        return {'text': '', 'pairs': [], 'n_views': len(images)}
    try:
        out = await synthesizer.pose_facts.remote(input_images=images, forward_key=key)
        facts = {'error': out.err['msg']} if out.err else (out.result or {})
        if get_config().omit_camera_context:
            facts = {**facts, 'text': '', 'camera_context_omitted': True}
        return facts
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}
