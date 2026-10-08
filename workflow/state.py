from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from tools.apis import AgentContext


class AgentState(TypedDict):
    """
    The state LangGraph threads through the pipeline nodes.

    Attributes:
        session_id (Optional[str]): Unique identifier of the sample being solved, used
            for the per-sample trace directory.
        messages (Annotated[...]): Short log messages emitted by the nodes.
        workspace (Dict[str, AgentContext]): Named values shared between nodes: the
            instruction and input images, the accumulated plan ('planner_output',
            a PlannerOutput written by s1_analyst / s2_pose_planner), and the final
            answer + trace record written by s4_solver.
    """
    session_id: Optional[str]

    messages: Annotated[List[AnyMessage], add_messages]
    workspace: Dict[str, AgentContext]
