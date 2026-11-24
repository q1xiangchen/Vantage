"""Expose view synthesis as an agent tool."""

from dataclasses import dataclass, field
from typing import Any, List

from PIL import Image
from ray import serve

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext


__ALL__ = ['ViewSynthesizer', 'SynthesizedView']


@dataclass
class SynthesizedView(AgentContext):
    """Store the rendered view and synthesis diagnostics."""
    image: Image.Image
    diag: dict = field(default_factory=dict)

    def to_message_content(self) -> str:
        return 'Synthesized 1 view.'


@serve.deployment
class ViewSynthesizer(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 12.0
    # One replica: the GPU headroom next to vLLM fits only one G3T peak.
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 1

    def __init__(self) -> None:
        super().__init__()
        from tools.apis.g3t_backend import G3TViewSynthesizer
        self._g3t = G3TViewSynthesizer(device='cuda')

    @AgentTool.document_output_class(SynthesizedView)
    async def synthesize(
        self,
        input_images: List[Image.Image],
        target_pose: Any,
        pivot_mode: str = 'ahead',
        anchor_index: int = 0,
        forward_key: str = '',
    ) -> AgentToolOutput:
        """Render a view at a target pose relative to the selected reference camera."""
        if not input_images:
            return self.error(msg='ViewSynthesizer.synthesize requires at least one input image.')
        if target_pose is None:
            return self.error(
                msg='ViewSynthesizer.synthesize requires an explicit target_pose '
                    '(the model decides the 6-DoF); there is no default motion.')
        try:
            image, diag = self._g3t.synthesize(
                input_images, target_pose=target_pose, pivot_mode=pivot_mode,
                anchor_index=anchor_index, forward_key=forward_key)
        except Exception as e:
            return self.error(msg=f'View synthesis failed: {e}')
        return self.success(result=SynthesizedView(image=image, diag=diag))

    async def pose_facts(self, input_images: List[Image.Image], forward_key: str = ''):
        """Return verbal camera relations and optionally retain the forward."""
        try:
            return self.success(self._g3t.pose_facts(
                input_images, forward_key=forward_key))
        except Exception as e:
            return self.error(msg=f'Pose guide failed: {e}')

    async def release_forward(self, forward_key: str):
        """Release a sample cache entry when planning or answering has finished."""
        getattr(self._g3t, '_fwd_cache', {}).pop(forward_key, None)
