"""View synthesis tool: render one new view of the scene at a target camera pose.

The input views need no camera poses. The target pose is a 4x4 c2w (OpenCV)
relative to the first input view (identity reproduces it). Backend: G3T
reconstruction (g3t_backend.py) + point reprojection (reprojection.py).
"""

from dataclasses import dataclass, field
from typing import Any, List

from PIL import Image
from ray import serve

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext


__ALL__ = ['ViewSynthesizer', 'SynthesizedView']


@dataclass
class SynthesizedView(AgentContext):
    """A synthesized view.
    Attributes:
        image (Image.Image): The rendered RGB view.
        diag (dict): Synthesis diagnostics, saved to the trace.
    """
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
    ) -> AgentToolOutput:
        """
        Synthesize one view of the scene at a target camera pose.

        Args:
            input_images (List[Image.Image]): The input views; the target pose is
                relative to the first one.
            target_pose: 4x4 c2w (OpenCV) relative to the first input view
                (list / numpy array / tensor). Required.
            pivot_mode (str): 'ahead' (rotate in place), 'centroid' (orbit the
                scene centre) or 'elevation' (rise/drop with the heading kept).
        """
        if not input_images:
            return self.error(msg='ViewSynthesizer.synthesize requires at least one input image.')
        if target_pose is None:
            return self.error(
                msg='ViewSynthesizer.synthesize requires an explicit target_pose '
                    '(the model decides the 6-DoF); there is no default motion.')
        try:
            image, diag = self._g3t.synthesize(
                input_images, target_pose=target_pose, pivot_mode=pivot_mode)
        except Exception as e:
            return self.error(msg=f'View synthesis failed: {e}')
        return self.success(result=SynthesizedView(image=image, diag=diag))
