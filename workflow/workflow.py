import asyncio
from collections import OrderedDict
from graphlib import TopologicalSorter
import inspect
import os
import torch
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from PIL import Image
import ray
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.handle import DeploymentHandle
import shortuuid

from tools.apis import (
    AGENT_TOOL_REGISTRY,
    AgentContext,
    FinalAnswer, 
    InputBBoxes2D,
    InputImages, 
    Instruction,
)
from workflow.config import MODE_VANTAGE, get_config
from workflow.logging import AgentLogger
from workflow.nodes import (
    S1Analyst,
    S2PosePlanner,
    S4Solver,
    after_s1,
)
from workflow.state import AgentState
from workflow.utils.cuda_utils import get_total_vram_gb
from workflow.utils.deps_utils import discover_dependencies, extract_dependency


class AgentWorkflow:
    
    def __init__(self):
        self.config = get_config()
        self.logger = AgentLogger()

        self.setup_ray()
        self.tool_configs = self.auto_configure()
        self.tool_handles = self.setup_serve()
        self.graph_nodes = self.setup_graph_nodes()
        self.graph = self.build_graph()
    
    def setup_ray(self):
        uid = shortuuid.ShortUUID().random(8)
        base = os.environ.get('RAY_TMPDIR') or '/tmp'
        logs_dir = os.path.join(base, f'ray_{os.environ.get("USER", "user")}_{uid}')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        if not ray.is_initialized():
            log_rotation = str(1000 * 1024 * 1024)  # 1G
            os.environ['RAY_ROTATION_MAX_BYTES'] = log_rotation
            os.environ['RAY_ROTATION_BACKUP_COUNT'] = '10'
            ray.init(_temp_dir=logs_dir)

        # Ray Serve's default HTTP proxy binds 127.0.0.1:8000, but this project
        # only ever calls tools via `serve.get_app_handle` (in-process handles) —
        # no external HTTP client hits 8000. On shared HPC nodes that port is
        # sometimes already occupied, which fails the proxy readiness check and
        # crashes the agent. Disable the proxy entirely.
        serve.start(proxy_location='Disabled')

    def auto_configure(self):
        full_dependency_graph = discover_dependencies()

        all_tools_to_deploy = self._get_all_required_tools(full_dependency_graph)
        ts = TopologicalSorter()
        for tool in all_tools_to_deploy:
            deps = full_dependency_graph.get(tool, [])
            dep_list = deps[1:] if isinstance(deps, Tuple) else []
            ts.add(tool, *dep_list)
        init_order = list(ts.static_order())

        print()
        total_vram_gb = get_total_vram_gb()
        tool_configs = OrderedDict()
        min_cost_cpu, min_cost_gpu = 0, 0
        for tool_name in init_order:
            tool_class = AGENT_TOOL_REGISTRY[tool_name]
            original_class = tool_class.func_or_class
            tool_config = {}

            # 1. Compute resource (CPU + GPU)
            num_cpus = getattr(original_class, 'CPU_CONSUMED', 1)
            ray_actor_options = {'num_cpus': num_cpus}
            vram_consumed = getattr(original_class, 'VRAM_CONSUMED', None)
            if vram_consumed:
                num_gpus = round(vram_consumed / total_vram_gb, 3)
                if 0 < num_gpus < 0.01:
                    num_gpus = 0.01
                ray_actor_options['num_gpus'] = num_gpus
            tool_config['ray_actor_options'] = ray_actor_options
            
            # 2. Tool repliacs or Autoscaling
            min_replicas = getattr(original_class, 'AUTOSCALING_MIN_REPLICAS', None)
            max_replicas = getattr(original_class, 'AUTOSCALING_MAX_REPLICAS', None)

            if tool_name == 'CoTReasoner':
                tool_config['num_replicas'] = self.config.concurrency
            elif min_replicas is None and max_replicas is None:
                tool_config['num_replicas'] = 1
            elif min_replicas == max_replicas:
                tool_config['num_replicas'] = min_replicas
            else:
                if vram_consumed is None:  # CPU Tool
                    downscale_delay_s, upscale_delay_s = 15.0, 5.0
                else:  # GPU Tool
                    downscale_delay_s, upscale_delay_s = 60.0, 30.0

                tool_config['autoscaling_config'] = AutoscalingConfig(
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    downscale_delay_s=downscale_delay_s,
                    upscale_delay_s=upscale_delay_s,
                )

            # 3. Tool execution time
            tool_config['graceful_shutdown_timeout_s'] = 300.0
            
            tool_configs[tool_name] = tool_config
            summary = f'[Tool Deployed] {tool_name} (num_cpus={tool_config["ray_actor_options"]["num_cpus"]}'
            if vram_consumed is not None:
                summary += f', num_gpus={tool_config["ray_actor_options"]["num_gpus"]}'
            if 'num_replicas' in tool_config:
                summary += f', num_replicas={tool_config["num_replicas"]}'
            if 'autoscaling_config' in tool_config:
                summary += f', min_replicas={tool_config["autoscaling_config"].min_replicas}'
                summary += f', max_replicas={tool_config["autoscaling_config"].max_replicas}'
            summary += ')'
            print(summary)

            tool_min_replicas = tool_config['num_replicas'] if 'num_replicas' in tool_config \
                else tool_config['autoscaling_config'].min_replicas
            min_cost_cpu += tool_min_replicas * num_cpus
            if vram_consumed is not None:
                min_cost_gpu += tool_min_replicas * num_gpus
        
        print(f'[Info] Total CPUs (min) Consumed: {min_cost_cpu}, Total GPUs Consumed: {min_cost_gpu}')

        cluster_resources = ray.cluster_resources()
        available_cpu = cluster_resources.get('CPU', 0)
        available_gpu = cluster_resources.get('GPU', 0)

        if min_cost_cpu > available_cpu:
            raise RuntimeError(
                f'Insufficient CPU resources. Minimum required: {min_cost_cpu:.2f}, '
                f'Available in cluster: {available_cpu}'
            )
        if min_cost_gpu > available_gpu:
            raise RuntimeError(
                f'Insufficient GPU resources. Minimum required: {min_cost_gpu:.2f}, '
                f'Available in cluster: {available_gpu}'
            )
        
        print(f'[Info] Resource check passed. (Available: {available_cpu} CPUs, {available_gpu} GPUs)\n')
        return tool_configs

    def setup_serve(self) -> Dict[str, DeploymentHandle]:
        tool_handles: Dict[str, DeploymentHandle] = {}
        for tool_name, tool_config in self.tool_configs.items():
            tool_class = AGENT_TOOL_REGISTRY[tool_name]
            if not isinstance(tool_class, serve.Deployment):
                raise RuntimeError(f'Expected tool_class to be "serve.Deployment", but got {type(tool_class)}')

            tool_class = tool_class.options(**tool_config)

            init_sign = inspect.signature(tool_class.func_or_class.__init__)
            init_args = {}
            for param in init_sign.parameters.values():
                if param.name == 'self' or param.annotation is inspect.Parameter.empty:
                    continue

                # handle optional dependency
                if isinstance(param.annotation, str) and param.annotation.startswith('Optional'):
                    _, dependency_class = extract_dependency(param.annotation)
                    dep_name = dependency_class.name
                else:
                    dep_name = getattr(param.annotation, 'name', None)

                if dep_name in tool_handles:
                    init_args[param.name] = tool_handles[dep_name]
            tool_application = tool_class.bind(**init_args)

            serve.run(
                tool_application, 
                name=tool_name, 
                route_prefix=f'/{tool_name.lower()}'
            )
            tool_handles[tool_name] = serve.get_app_handle(tool_name)

        return tool_handles

    def _get_all_required_tools(
        self, 
        full_dependency_graph: Dict[str, str | Tuple]
    ) -> set:
        # CoTReasoner is always needed; ViewSynthesizer only in vantage mode.
        # Their transitive deps are pulled in below.
        top_level_tools = {'CoTReasoner'}
        if self.config.mode == MODE_VANTAGE:
            top_level_tools.add('ViewSynthesizer')

        all_reqs = set()
        def find_deps_recursively(tool):
            if tool in all_reqs:
                return
            
            all_reqs.add(tool)
            deps = full_dependency_graph.get(tool, [])
            dep_list = deps[1:] if isinstance(deps, Tuple) else []
            for dep in dep_list:
                find_deps_recursively(dep)

        for tool in top_level_tools:
            find_deps_recursively(tool)
        return all_reqs

    def setup_graph_nodes(self) -> Dict:
        reasoner = self.tool_handles['CoTReasoner']
        nodes = {
            's1_analyst': S1Analyst(reasoner, self.logger),
            's2_pose_planner': S2PosePlanner(reasoner, self.logger),
            's4_solver': S4Solver(
                reasoner=reasoner,
                logger=self.logger,
                view_synthesizer=self.tool_handles.get('ViewSynthesizer'),
            ),
        }

        return nodes

    def build_graph(self, checkpointer=None) -> StateGraph:
        graph = StateGraph(AgentState)

        # s4_solver is the single node that writes 'final_answer'.
        graph.add_node('s1_analyst', self.graph_nodes['s1_analyst'].run)
        graph.add_node('s2_pose_planner', self.graph_nodes['s2_pose_planner'].run)
        graph.add_node('s4_solver', self.graph_nodes['s4_solver'].run)

        graph.set_entry_point('s1_analyst')
        graph.add_conditional_edges('s1_analyst', after_s1)
        graph.add_edge('s2_pose_planner', 's4_solver')
        graph.add_edge('s4_solver', END)

        return graph.compile(checkpointer=checkpointer)

    async def arun(
        self,
        instruction: str,
        images: Optional[str | Image.Image | List[str | Image.Image]] = None,
        bbox2d: Optional[torch.Tensor] = None,
        answer: Optional[str] = None,
        session_id: Optional[str] = None,
        answer_format: str = '',
    ) -> AgentState:
        if not session_id:
            session_id = shortuuid.ShortUUID().random(8)
        self.logger.get_session_dir(session_id)

        run_config = {'configurable': {'thread_id': session_id}}
        initial_workspace = {
            'instruction': Instruction(text=instruction, answer_format=answer_format)}

        if images:
            if not isinstance(images, List):
                images = [images]

            if isinstance(images[0], Image.Image):
                input_images = InputImages(images=images)
            else:
                refs = [
                    self.tool_handles['ImageLoader'].load_image.remote(source)
                    for source in images
                ]
                results = await asyncio.gather(*refs)
                
                loaded_images = []
                for result in results:
                    if result.err:
                        raise RuntimeError(result.err['msg'])
                    loaded_images.append(result.result)
                
                input_images = InputImages(_sources=images, images=loaded_images)
            
            initial_workspace['input_images'] = input_images
        
        if bbox2d is not None:
            initial_workspace['input_bbox2d'] = InputBBoxes2D(bbox2d=bbox2d)

        initial_messages = HumanMessage(content=instruction)

        initial_state: AgentState = {
            'workspace': initial_workspace,
            'messages': [initial_messages],
            'session_id': session_id,
        }
        final_state = await self.graph.ainvoke(initial_state, config=run_config)

        return final_state
    
    def get_final_answer(
        self, 
        final_state: AgentState
    ) -> Tuple[AgentContext, Optional[str]]:
        workspace = final_state.get('workspace', {})
        if 'final_answer' not in workspace:
            raise KeyError(
                'Could not find "final_answer" in the final workspace. '
                'The agent workflow may not have completed successfully.'
            )
        
        final_answer: FinalAnswer = workspace['final_answer']
        numeric_result = final_answer.result
        summary = final_answer.natural_language_summary
        return (numeric_result, summary)

    def shutdown(self):
        serve.shutdown()
        ray.shutdown()
