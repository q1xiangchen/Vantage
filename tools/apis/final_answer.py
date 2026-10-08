from dataclasses import dataclass
from typing import Any, Optional

from tools.apis.base import AgentContext

__ALL__ = ['FinalAnswer']


@dataclass
class FinalAnswer(AgentContext):
    result: AgentContext | Any = None
    natural_language_summary: Optional[str] = None

    def to_message_content(self):
        if self.natural_language_summary:
            return self.natural_language_summary
        
        if isinstance(self.result, AgentContext) and hasattr(self.result, 'to_message_content'):
            summary = self.result.to_message_content()
            return f'Final answer computed. Result summary: {summary}'
        return f'Final answer computed. Result ({type(self.result).__name__}): {self.result}.'
