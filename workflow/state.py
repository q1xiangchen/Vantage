from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from tools.apis import AgentContext


class AgentState(TypedDict):
    """Carry conversation messages and workspace data between pipeline nodes."""
    session_id: Optional[str]

    messages: Annotated[List[AnyMessage], add_messages]
    workspace: Dict[str, AgentContext]
