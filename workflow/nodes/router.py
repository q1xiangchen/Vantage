from workflow.config import MODE_BASELINE, get_config
from workflow.state import AgentState


# The pipeline is s1_analyst -> s2_pose_planner -> s4_solver. Baseline runs s4 alone;
# an s1 failure skips s2 and degrades to a bare VQA in s4.


def after_s1(state: AgentState) -> str:
    if get_config().mode == MODE_BASELINE:
        return 's4_solver'
    po = state.get('workspace', {}).get('planner_output')
    if po is not None and po.analysis is not None:
        return 's2_pose_planner'
    return 's4_solver'
