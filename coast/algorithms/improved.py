import abc
import collections
import os
import queue
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
import pddlboat as pd
import symbolic
import traceback
import pybullet_tools as pbt
from coast.algorithms.base import Log
from coast.algorithms.binding import PrioritizedBindings
from coast.action import Action, ActionCall
from coast.algorithms.common import ordered_preimage
from coast.object import Object, StreamObject
from coast.algorithms import base, common
from coast.algorithms.common import COAST_METHODS
from coast.constraint import Constraint
from coast.formula import _P
from coast.stream import Stream, StreamConstraint, StreamInstance
from coast import utils
from coast.fastdownward import ConstraintError, write_feedback
import numpy as np
import glob


# ---------------------------------------------------------------------------
# VLM conversation reset
# ---------------------------------------------------------------------------

def _reset_vlm_conversation(exp: str) -> None:
    """Delete all VLM conversation artefacts so the next problem starts fresh.

    Removes:
      envs/{exp}/prompting/output/feedback_*.json
      envs/{exp}/prompting/output/constraints_*.py
      envs/{exp}/prompting/output/initial_prompt_*.txt
      envs/{exp}/prompting/output/dump.txt
      envs/{exp}/temp_constraints.py
    """
    output_dir = f"envs/{exp}/prompting/output"
    patterns = [
        f"{output_dir}/feedback_*.json",
        f"{output_dir}/constraints_*.py",
        f"{output_dir}/initial_prompt_*.txt",
        f"{output_dir}/VLMB_initial_prompt.txt",
        f"{output_dir}/VLMB_plan_*.txt",
        f"{output_dir}/dump.txt",
    ]
    removed = 0
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    temp_path = f"envs/{exp}/temp_constraints.py"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[improved] Reset VLM conversation for '{exp}': removed {removed} file(s)")


# ---------------------------------------------------------------------------
# Incremental constraint helpers
# ---------------------------------------------------------------------------

def _parse_pddl_proposition(prop: str) -> Tuple[str, List[str]]:
    """Parse a proposition into (predicate, [arg1, arg2, ...]).

    Handles both PDDL format   '(ontable b1 loc5)'
    and symbolic format         'ontable(b1, loc5)' / 'handempty()'.
    """
    s = prop.strip()
    # symbolic format: 'pred(a1, a2)' or 'handempty()'
    if "(" in s and not s.startswith("("):
        head = symbolic.parse_head(s)
        args = [a for a in symbolic.parse_args(s) if a]  # filter empty strings
        return (head, args)
    # PDDL format: '(pred a1 a2)'
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = s.split()
    return (parts[0], parts[1:]) if parts else ("", [])


def _write_incremental_constraints(
    exp: str,
    blocked_pairs: List[Tuple[str, List[str], List[Tuple[str, List[str]]]]],
) -> None:
    """Rewrite envs/{exp}/temp_constraints.py with block_state_action calls.

    Each element of *blocked_pairs* is
        (action_name, [arg1, ...], [(pred, [obj1, ...]), ...])
    representing a state-action pair that should be blocked.
    """
    path = f"envs/{exp}/temp_constraints.py"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines: List[str] = ["def generate_constraints(problem, domain, planner, pd):"]

    if not blocked_pairs:
        lines.append("    pass  # No blocked pairs yet")
    else:
        for idx, (action_name, action_args, state_props) in enumerate(blocked_pairs):
            lines.append("")
            lines.append(
                f"    # Blocked pair {idx + 1}: "
                f"{action_name}({', '.join(action_args)})"
            )
            lines.append(f'    _a{idx} = domain.get_action("{action_name}")')
            lines.append(f"    _assign{idx} = pd.Assignment()")
            lines.append(f"    _args{idx} = {action_args!r}")
            lines.append(
                f"    for _obj, _param in zip(_args{idx}, _a{idx}.parameters):"
            )
            lines.append(
                f"        _assign{idx}.assign(problem, _param.name, _obj)"
            )
            lines.append(f"    _st{idx} = pd.State(problem)")
            for pred_name, pred_args in state_props:
                if len(pred_args) == 0:
                    lines.append(
                        f'    _st{idx}.assign0("{pred_name}", True)'
                    )
                elif len(pred_args) == 1:
                    lines.append(
                        f'    _st{idx}.assign1("{pred_name}", '
                        f'"{pred_args[0]}", True)'
                    )
                else:
                    args_str = ", ".join(f'"{a}"' for a in pred_args)
                    lines.append(
                        f'    _st{idx}.assign_n("{pred_name}", '
                        f"[{args_str}], True)"
                    )
            lines.append(
                f"    planner.block_state_action(_a{idx}, _assign{idx}, _st{idx})"
            )

    lines.append("")
    lines.append("generate_constraints(problem, domain, planner, pd)")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(
        f"[incremental] Wrote {path} with {len(blocked_pairs)} blocked pair(s)"
    )


def _record_incremental_block(
    optimistic_plan: Sequence,
    failure_info: Optional[Dict[str, str]],
    blocked_pairs: List[Tuple[str, List[str], List[Tuple[str, List[str]]]]],
    exp: str,
) -> bool:
    """Parse failure_info, append a blocked pair, rewrite the constraints file.

    Returns True if a new pair was recorded, False otherwise.
    """
    if failure_info is None:
        print("[incremental] No failure_info; skipping constraint")
        return False

    task_step_str = failure_info.get("task_step", "unknown")
    task_action_str = failure_info.get("task_action", "unknown")
    if task_step_str == "unknown" or task_action_str == "unknown":
        print("[incremental] Cannot determine failed action; skipping constraint")
        return False

    task_step = int(task_step_str)
    if task_step < 1 or task_step > len(optimistic_plan) - 1:
        print(f"[incremental] Invalid task_step {task_step}; skipping constraint")
        return False

    # Pre-state is the state *before* the failed action.
    # optimistic_plan[0] = initial node (action="", state=S0)
    # optimistic_plan[k] = node after action k  (state=Sk)
    # Pre-state of action at task_step (1-indexed) = optimistic_plan[task_step - 1].state
    pre_state: Set[str] = optimistic_plan[task_step - 1].state

    action_name = symbolic.parse_head(task_action_str)
    action_args = list(symbolic.parse_args(task_action_str))
    state_props = [
        _parse_pddl_proposition(p)
        for p in sorted(pre_state)
    ]
    # Filter out empty/bad parses
    state_props = [(n, a) for n, a in state_props if n]

    blocked_pairs.append((action_name, action_args, state_props))
    _write_incremental_constraints(exp, blocked_pairs)
    print(
        f"[incremental] Blocked {action_name}({', '.join(action_args)}) "
        f"from state with {len(state_props)} atoms "
        f"(total blocked: {len(blocked_pairs)})"
    )
    return True
def infer_action_arguments(
    stream_state: Set[str],
    task_plan: Sequence[symbolic.PlannerNode],
    geometric_actions: Dict[str, Action],
    objects: Mapping[str, Object],
) -> Tuple[List[ActionCall], List[Dict[str, None]], Dict[str, Object]]:
    state = {s: None for s in stream_state}
    new_objects = dict(objects)
    action_calls = []
    states = [state]
    # print("stream state:",stream_state)
    # print("objects:", objects)
    for t, node in enumerate(task_plan[1:]):
        action_call = node.action
        action = geometric_actions[symbolic.parse_head(action_call)]
        args_map = action.map_inputs_to_objects(action_call, state)

        for output in action.outputs:
            new_output = StreamObject(f"{output.name[1:]}_{t+1}", output.type)
            args_map[output.name] = new_output.name
            new_objects[new_output.name] = new_output

        action_call = ActionCall(action, args_map)
        action_calls.append(action_call)
        # print("what is the type of the action_plan")
        state = action_call.apply(state, new_objects)
        states.append(state)

    return action_calls, states, new_objects


def sort_stream_plan_by_descendants(
    stream_plan: List[StreamInstance],
) -> List[StreamInstance]:
    def count_descendants(
        stream_instance: StreamInstance,
        dependency_graph: Dict[str, List[StreamInstance]],
        num_descendants: Dict[StreamInstance, int],
    ) -> int:
        try:
            return num_descendants[stream_instance]
        except KeyError:
            children = set(
                itertools.chain.from_iterable(
                    dependency_graph[arg.name]
                    for arg in stream_instance.outputs.values()
                )
            )
            num_descendants[stream_instance] = sum(
                1 + count_descendants(child, dependency_graph, num_descendants)
                for child in children
            )
        return num_descendants[stream_instance]

    # Compute dependency graph.
    dependency_graph: Dict[str, List[StreamInstance]] = collections.defaultdict(list)
    for stream_instance in stream_plan:
        all_inputs = itertools.chain(
            stream_instance.inputs.values(), stream_instance.fluent_inputs.values()
        )
        for arg in all_inputs:
            dependency_graph[arg.name].append(stream_instance)

    num_descendants: Dict[StreamInstance, int] = {}
    sorted_stream_plan = sorted(
        (
            -count_descendants(stream_instance, dependency_graph, num_descendants),
            idx_plan,
            stream_instance,
        )
        for idx_plan, stream_instance in enumerate(stream_plan)
    )

    stream_plan = [stream_instance[2] for stream_instance in sorted_stream_plan]

    return stream_plan


def sort_stream_plan(
    stream_plan: List[StreamInstance],
    key: Callable[[StreamInstance], Any],
    idx_start: int = 0,
) -> List[StreamInstance]:
    ancestors_map: Dict[StreamInstance, Set[StreamInstance]] = {}

    def get_ancestors(
        stream_instance: StreamInstance, output_map: Dict[str, StreamInstance]
    ) -> Set[StreamInstance]:
        try:
            return ancestors_map[stream_instance]
        except KeyError:
            ancestors = set()
            all_inputs = itertools.chain(
                stream_instance.inputs.values(), stream_instance.fluent_inputs.values()
            )
            for arg in all_inputs:
                if not isinstance(arg, StreamObject):
                    continue
                parent = output_map[arg.name]
                ancestors |= get_ancestors(parent, output_map)
                ancestors.add(parent)

            ancestors_map[stream_instance] = ancestors
        return ancestors_map[stream_instance]

    def sorted_plan(
        stream_plan: List[StreamInstance],
        output_map: Dict[str, StreamInstance],
        key: Callable[[StreamInstance], Any],
        visited: Set[StreamInstance] = set(),
    ) -> List[StreamInstance]:
        if len(stream_plan) <= 1:
            return stream_plan
        # mina = -1
        # for s in stream_plan:
        #     cost = key(s)
        #     # print(s, min)
        #     # input(cost)
        #     if mina == -1 or cost < mina:
        #         mina = cost
        #         next_stream = s
        # print("Next stream", next_stream, key(next_stream))
        next_stream = min(stream_plan, key=key)
        # assert a == next_stream
        visited.add(next_stream)

        ancestors = get_ancestors(next_stream, output_map)
        ancestor_plan = [s for s in stream_plan if s not in visited and s in ancestors]
        remaining_plan = [
            s for s in stream_plan if s not in visited and s not in ancestors
        ]

        return (
            sorted_plan(ancestor_plan, output_map, key, visited)
            + [next_stream]
            + sorted_plan(remaining_plan, output_map, key, visited)
        )

    # Compute ancestor graph.
    output_map: Dict[str, StreamInstance] = {}
    for stream_instance in stream_plan:
        for arg in stream_instance.outputs.values():
            # print(arg.name, output_map)
            assert arg.name not in output_map
            output_map[arg.name] = stream_instance
    # import pdb; pdb.set_trace()
    sorted_stream_plan = stream_plan[:idx_start] + sorted_plan(
        stream_plan[idx_start:], output_map, key
    )

    return sorted_stream_plan


def sort_stream_plan_by_inputs(
    stream_plan: List[StreamInstance], idx_start: int = 0
) -> List[StreamInstance]:
    cache: Dict[str, int] = {}

    def num_inputs(stream_instance: StreamInstance) -> int:
        key = str(stream_instance)
        try:
            return cache[key]
        except KeyError:

            cache[key] = len( # -1
                [
                    arg
                    for arg in itertools.chain(
                        stream_instance.inputs, stream_instance.fluent_inputs
                    )
                    #if isinstance(arg, StreamObject)
                ]
            )  # - len(s.outputs),

            return cache[key]
    return sort_stream_plan(stream_plan, num_inputs, idx_start)


def sort_stream_plan_by_failures(
    stream_plan: List[StreamInstance], log: Log, idx_start: int = 0
) -> List[StreamInstance]:
    return sort_stream_plan(stream_plan, lambda s: log.success_rate(s), idx_start)


def sort_stream_plan_by_expected_time(
    stream_plan: List[StreamInstance], log: Log, idx_start: int = 0
) -> List[StreamInstance]:
    cache: Dict[str, Tuple[bool, float]] = {}

    def expected_time(stream_instance: StreamInstance) -> Tuple[bool, float]:
        key = stream_instance.stream.name
        try:
            return cache[key]
        except KeyError:
            p = log._success_rate(key)
            cache[key] = (p == 1.0, p * log._average_time(key))
            return cache[key]

    return sort_stream_plan(stream_plan, expected_time, idx_start)


def retrace(
    stream_state: Set[str],
    task_plan: Sequence[symbolic.PlannerNode],
    actions_map: Dict[str, Action],
    streams_map: Dict[str, Stream],
    predicates_map: Dict[str, _P],
    objects: Mapping[str, Object],
    sort_streams: bool = True
) -> Tuple[List[StreamInstance], List[ActionCall]]:
    action_plan, states, new_objects = infer_action_arguments(
        stream_state, task_plan, actions_map, objects
    )



    stream_plan = []
    for action_call, pre_state, post_state in zip(action_plan, states[:-1], states[1:]):
        state = {prop: None for prop in pre_state if prop in post_state}
        certified_facts = action_call.get_certified_facts(
            state, list(new_objects.values())
        )
        for fact in certified_facts:
            pred, args = symbolic.parse_proposition(fact)
            stream = streams_map[pred]
            inputs = {param.name[1:]: param for param in stream.inputs}
            outputs = {
                param.name[1:]: param for param in stream.outputs
            }  # TODO: Remove ? from param name
            # print(f"[RETRACE] Created stream instance {action_call} with inputs={inputs} outputs={outputs}")
            # breakpoint()
            fact_template = stream.certified
            # input(f"{fact_template.args} {args}")
            assert len(fact_template.args) == len(args)
            for param, arg in zip(fact_template.args, args):
                param = param[1:]  # TODO: Remove ? from param name
                if param in inputs:
                    inputs[param] = new_objects.get(arg, arg)
                elif param in outputs:
                    outputs[param] = new_objects[arg]
            fluents = {
                pred: [
                    {
                        param: new_objects[arg]
                        for param, arg in zip(
                            predicates_map[pred].args, symbolic.parse_args(prop)
                        )
                    }
                    for prop in state
                    if symbolic.parse_head(prop) == pred
                ]
                for pred in stream.fluents
            }

            stream_instance = stream.create_instance(
                inputs=inputs,
                outputs=outputs,
                action_call=action_call,
                fluents=fluents,
            )
            stream_plan.append(stream_instance)
        # breakpoint()
    if sort_streams:
        sorted_stream_plan = sort_stream_plan_by_inputs(stream_plan)
    else:
        sorted_stream_plan = stream_plan
    return sorted_stream_plan, action_plan

def rep_constraints(constraints):
    s = ""
    for i in range(len(constraints)):
        s += str(constraints[i])
    # print("constrints :",s)
    # if len(constraints)>0:
    #     breakpoint()
    return s
class Improved(base.Algorithm, abc.ABC):
    # Persists prompt numbering across multiple problem instances in one process.
    _session_prompt_number_by_exp: Dict[str, int] = {}
    # Tracks whether the last problem for each exp failed (timeout / max iters).
    # When True the next problem resets the VLM conversation.
    _last_problem_failed_by_exp: Dict[str, bool] = {}

    def __init__(
        self,
        max_level: int,
        search_sample_ratio: float,
        timeout: float,
        use_cache: bool = False,
    ):
        super().__init__(max_level=max_level)
        self.search_sample_ratio = search_sample_ratio
        self.timeout = timeout
        self.use_cache = use_cache


    def plan(
        self,
        cfg: Any,
        pddl: symbolic.Pddl,
        streams: Sequence[Stream],
        actions: Sequence[Action],
        geometric_predicates: Sequence[_P],
        stream_state: Set[str],
        objects: Sequence[Object],
        world: Any,
        constraint_cls: Type[Constraint],
        sort_streams: bool = True
    ) -> base.Plan:
        objects_map = {obj.name: obj for obj in objects}
        actions_map = {type(action).__name__.lower(): action for action in actions}
        streams_map = {stream.certified.predicate: stream for stream in streams}
        predicates_map = {p.predicate: p for p in geometric_predicates}
        exp_name = str(cfg.get("exp", ""))

        constraints: List[Constraint] = []
        constrained_pddl = pddl
        # print("what are those objects?",type(objects))

        log = Log()
        timer = utils.Timer()
        timer.tic("plan")
        # Store queue of successful task plan and current constraints
        plan_queue: queue.PriorityQueue[Tuple[
                Union[List[symbolic.PlannerNode], symbolic.Pddl, List[StreamConstraint]]
            ]] = queue.PriorityQueue()
        plan_queue.put((0, 0, timer.toc("plan"), [], pddl, constraints[:]))
        if self.use_cache:
            cache: Dict = {}
        else:
            cache = None
        constraint_map = collections.defaultdict(int)
        fail = True
        failed_actions=[None]
        planner_condition_list=[]
        #### START Dynamically overwrite the PDDL for block world, REMEMBER TO EDIT THIS PART IN THE FUTURE

        #
        # domain_z3 = pd.parse_domain("envs/shelf/domain_coast_block.pddl")
        # problem_z3 = pd.parse_problem(domain_z3, "envs/shelf/problem_coast_block.pddl")
        # plannerz3 = pd.Z3Planner(problem_z3, pd.Z3Planner.Options())
        # planner_condition_list.append(plannerz3)
        #### END

        # Number of consecutive failed outer-loop iterations used to gate VRM re-prompts.
        failed_iteration_count = 0
        reprompt_failure_threshold = max(1, int(cfg.get("reprompt_failure_threshold", 1)))
        # VLMB produces problem-specific action plans (not generalisable
        # constraints like zoast), so we ALWAYS reset between problems.
        # For zoast we only reset when the previous problem failed.
        method = cfg.get("method", "")
        if method == "VLMB":
            _reset_vlm_conversation(exp_name)
            cfg["_prompt_number"] = 0
            self._session_prompt_number_by_exp[exp_name] = 0
        elif self._last_problem_failed_by_exp.get(exp_name, False):
            _reset_vlm_conversation(exp_name)
            cfg["_prompt_number"] = 0
            self._session_prompt_number_by_exp[exp_name] = 0
        # Keep prompt numbering continuous across successful problems
        # (only meaningful for zoast; VLMB always starts at 0 above).
        if "_prompt_number" not in cfg:
            cfg["_prompt_number"] = int(
                self._session_prompt_number_by_exp.get(exp_name, 0)
            )
        else:
            cfg["_prompt_number"] = int(cfg["_prompt_number"])
        self._session_prompt_number_by_exp[exp_name] = int(cfg["_prompt_number"])

        # Incremental method: maintain a list of blocked (state, action) pairs
        # and reset the constraints file at the start of each problem.
        is_incremental = cfg.get("method") == "incremental"
        # zoast also uses incremental-style state-action blocks between reprompts
        # so that each iteration within the same prompt window gets a different plan.
        uses_incremental_blocks = cfg.get("method") in ("incremental", "zoast")
        incremental_blocked_pairs: List[
            Tuple[str, List[str], List[Tuple[str, List[str]]]]
        ] = []
        if uses_incremental_blocks:
            _write_incremental_constraints(exp_name, incremental_blocked_pairs)
        if is_incremental:
            # Iterative deepening: start at a configurable horizon and grow
            # when the Z3 solver becomes UNSAT (all plans at this depth blocked).
            incremental_horizon = int(cfg.get("incremental_initial_horizon", 1))
            cfg["_incremental_horizon"] = incremental_horizon
            print(f"[incremental] Starting with horizon = {incremental_horizon}")

        for idx_iter in itertools.count():
            # Re-populate the queue only for methods that need it (incremental
            # uses iterative deepening which clears constraints and must restart;
            # zoast may need to retry after clearing incremental blocks).
            # For 'coast', an empty queue means the constraint space is exhausted
            # and the algorithm should terminate — matching original COAST.
            if plan_queue.qsize() <= 0 and cfg.get('method') in ('incremental', 'zoast'):
                plan_queue.put((0, 0, timer.toc("plan"), [], pddl, constraints[:]))

            action_plan: Optional[List[ActionCall]] = None
            maybe_bindings: Optional[
                Union[Mapping[str, Object], StreamConstraint]
            ] = None
            log.set_iteration(idx_iter)

            if timer.toc("plan") > self.timeout:
                print("[improved.plan] Failure: timeout")
                break

            #VIZ-COAST ADDED
            if cfg['max_iterations'] != 0 and idx_iter >= cfg['max_iterations']:
                print("[improved.plan] Failure: max iterations")
                break

            print(f"\n[improved.plan] ========== Iter: {idx_iter} ==========")
            count, _, _, optimistic_plan, new_pddl, constraints = plan_queue.get()
            print(f"[improved.plan] Dequeing task plan count: {count} len_constraints: {len(constraints)} empty? {plan_queue.empty()}")
            reprompted_this_iter = False

            #VIZ-COAST ADDED -- don't use constraints if using vizcoast
            if cfg['method'] in COAST_METHODS:
                constraint_map[rep_constraints(constraints)] += 1
                constrained_pddl = new_pddl
                if len(constraints) > 0:
                    constrained_pddl = Constraint.constrain_pddl(pddl, constraints)

            print("[improved.plan] State:", constrained_pddl.initial_state)

            exp = cfg.get('exp', '')

            # TASK PLANNING #############################################
            with log.profile("search"):

                try:
                    optimistic_plan, new_pddl = common.search(
                        cfg, failed_iteration_count, constrained_pddl, constrained_pddl.initial_state,
                        use_symbolic=False, failed_last_action=failed_actions,planner_condition_list=planner_condition_list, log=log,
                        iteration=idx_iter)
                    print("returned to improved.py")
                    reprompted_this_iter = bool(cfg.pop("_reprompted_this_iteration", False))
                except ConstraintError as e:
                    reprompted_this_iter = bool(cfg.pop("_reprompted_this_iteration", False))
                    print('Constraint bug in search:', e)
                    traceback.print_exc()
                    if cfg.get('method') in ('zoast', 'VLMB'):
                        write_feedback(exp, "constraint_bug",
                                       prompt_number=cfg["_prompt_number"],
                                       error=str(e))
                        # Constraint bugs should trigger an immediate re-prompt on the
                        # next iteration, independent of the normal threshold gate.
                        failed_iteration_count = reprompt_failure_threshold
                        print(
                            "[improved.plan] Constraint bug detected; forcing immediate "
                            f"re-prompt (threshold={reprompt_failure_threshold})."
                        )
                    else:
                        if reprompted_this_iter:
                            print("[improved.plan] Re-prompt occurred; resetting failed_iteration_count.")
                            failed_iteration_count = 0
                        failed_iteration_count += 1
                    # Re-enqueue so the next iteration doesn't block on an empty queue.
                    if plan_queue.empty():
                        plan_queue.put((count, idx_iter, timer.toc("plan"), [], constrained_pddl, constraints[:]))
                    continue
                except Exception as e:
                    reprompted_this_iter = bool(cfg.pop("_reprompted_this_iteration", False))
                    print('ERROR IN SEARCH:', e)
                    traceback.print_exc()
                    if cfg.get('method') in ('zoast', 'VLMB'):
                        write_feedback(exp, "search_error",
                                       prompt_number=cfg["_prompt_number"],
                                       error=str(e))
                        failed_iteration_count = reprompt_failure_threshold
                        print(
                            "[improved.plan] Search error detected; forcing immediate "
                            f"re-prompt (threshold={reprompt_failure_threshold})."
                        )
                    else:
                        if reprompted_this_iter:
                            print("[improved.plan] Re-prompt occurred; resetting failed_iteration_count.")
                            failed_iteration_count = 0
                        failed_iteration_count += 1
                    # Re-enqueue so the next iteration doesn't block on an empty queue.
                    if plan_queue.empty():
                        plan_queue.put((count, idx_iter, timer.toc("plan"), [], constrained_pddl, constraints[:]))
                    continue
                finally:
                    # task_plan() may advance _prompt_number during prompt_vlm.
                    self._session_prompt_number_by_exp[exp_name] = int(
                        cfg.get("_prompt_number", 0)
                    )

            print(f"[improved.plan] Search: {log.get_last('search')}s")
            if reprompted_this_iter:
                print("[improved.plan] Re-prompt occurred; resetting failed_iteration_count.")
                failed_iteration_count = 0
                # New VLM constraints → clear incremental blocks so the fresh
                # constraints are explored without stale no-goods.
                if uses_incremental_blocks and incremental_blocked_pairs:
                    n_cleared = len(incremental_blocked_pairs)
                    incremental_blocked_pairs.clear()
                    _write_incremental_constraints(exp_name, incremental_blocked_pairs)
                    print(f"[improved.plan] Cleared {n_cleared} incremental block(s) after re-prompt")
            if optimistic_plan is None or (optimistic_plan is not None and len(optimistic_plan) <= 1):
                no_plan_label = "no valid plan found" if optimistic_plan is None else "empty task plan (no actions)"
                print(f"[improved.plan] Search Failed: {no_plan_label}")
                log.num_fail_task_plans += 1

                if is_incremental:
                    # Iterative deepening: UNSAT means all plans at the
                    # current horizon have been blocked.  Clear the
                    # constraint list and increment the horizon.
                    old_h = int(cfg.get("_incremental_horizon", 1))
                    new_h = old_h + 1
                    n_cleared = len(incremental_blocked_pairs)
                    incremental_blocked_pairs.clear()
                    _write_incremental_constraints(exp_name, incremental_blocked_pairs)
                    cfg["_incremental_horizon"] = new_h
                    print(
                        f"[incremental] UNSAT at horizon {old_h} — "
                        f"cleared {n_cleared} blocked pair(s), "
                        f"incremented horizon to {new_h}"
                    )
                    continue

                if cfg.get('method') in ('zoast', 'VLMB'):
                    write_feedback(exp, "no_plan_found",
                                   prompt_number=cfg["_prompt_number"])
                    # Incremental blocks exhausted the plan space under current
                    # VLM constraints — clear them so the reprompt starts fresh.
                    if cfg.get('method') == 'zoast' and incremental_blocked_pairs:
                        n_cleared = len(incremental_blocked_pairs)
                        incremental_blocked_pairs.clear()
                        _write_incremental_constraints(exp_name, incremental_blocked_pairs)
                        print(f"[improved.plan] Cleared {n_cleared} incremental block(s) (UNSAT under zoast)")
                    failed_iteration_count = reprompt_failure_threshold
                    print(
                        "[improved.plan] No plan found; forcing immediate "
                        f"re-prompt (threshold={reprompt_failure_threshold})."
                    )
                    # Re-enqueue the current state so the next iteration can
                    # actually run a reprompt instead of blocking on an empty queue.
                    plan_queue.put((count, idx_iter, timer.toc("plan"), [], constrained_pddl, constraints[:]))
                    continue
                if plan_queue.empty():
                    print("Empty Queue")
                    break
                else:
                    failed_iteration_count += 1
                    continue
            else:
                log.num_success_task_plans += 1

            plan_actions = [node.action for node in optimistic_plan[1:]]
            print(f"[improved.plan] Plan: {plan_actions}")   
            print(f"[improved.plan] plan length: {len(optimistic_plan)}")

            # MOTION PLANNING #############################################
            if len(optimistic_plan)>1:
                # RETRACE --------------------------------------
                print("[improved.plan] Starting retrace")
                with log.profile("retrace"):
                    stream_plan, action_plan = retrace(
                        stream_state=stream_state,
                        task_plan=optimistic_plan,
                        actions_map=actions_map,
                        streams_map=streams_map,
                        predicates_map=predicates_map,
                        objects=objects_map,
                        sort_streams=sort_streams
                    )
                    if sort_streams:
                        stream_plan = sort_stream_plan_by_expected_time(stream_plan, log)
                    for stream_instance in stream_plan:
                        stream_instance.clear_count()

                # PROCESS STREAMS --------------------------------------
                print("[improved.plan] Starting process streams")
                with log.profile("process"):
                    maybe_bindings, visited_bindings, failure_info = self.process_streams(
                        stream_plan=stream_plan,
                        optimistic_plan=optimistic_plan[1:],
                        action_plan=action_plan,
                        timeout=min(
                            (self.search_sample_ratio * log.compute_sum("search") - log.compute_sum("process")),
                            self.timeout - timer.toc("plan"),
                        ),
                        world=world,
                        bindings=objects_map,
                        pddl=new_pddl,
                        log=log,
                        cache=cache
                    )

                print(f"[improved.plan] Process: {log.get_last('process')}s")
                failure_detail = "The task plan could not be grounded to valid motion plans."
                if failure_info is not None:
                    failure_detail = (
                        "Stream/sample failure: "
                        f"stream='{failure_info.get('stream')}', "
                        f"task_step={failure_info.get('task_step')}, "
                        f"task_action='{failure_info.get('task_action')}'. "
                        # f"action_call='{failure_info.get('action_call')}'."
                    )
                # Add to queue but without the constraint
                plan_queue.put((constraint_map[rep_constraints(constraints)], -len(constraints), -timer.toc("plan"), optimistic_plan, new_pddl, constraints[:]))
                if (
                    isinstance(maybe_bindings, StreamConstraint)
                    and maybe_bindings is not None
                ):
                    constraint_type = maybe_bindings.constraint_type
                    constraints.append(
                        constraint_type(
                            pddl=new_pddl,
                            task_plan=optimistic_plan,
                            action_plan=action_plan,
                            stream_plan=stream_plan,
                            objects=objects,
                            visited_bindings=visited_bindings,
                            world=world,
                            constraint=maybe_bindings.constraint,
                        )
                    )
                    # Add plan, pddl, and new constraints to queue
                    plan_queue.put((constraint_map[rep_constraints(constraints)], -len(constraints), -timer.toc("plan"), optimistic_plan, new_pddl, constraints[:]))
                    # Refinement failed with a stream constraint — write feedback
                    if cfg.get('method') in ('zoast', 'VLMB'):
                        write_feedback(exp, "refinement_failed",
                                       prompt_number=cfg["_prompt_number"],
                                       plan=plan_actions,
                                        detail=f"{failure_detail} Motion planning found a collision constraint.")
                    if uses_incremental_blocks:
                        _record_incremental_block(
                            optimistic_plan, failure_info,
                            incremental_blocked_pairs, exp_name,
                        )
                    failed_iteration_count += 1
                    continue
                if maybe_bindings is not None:
                    self._last_problem_failed_by_exp[exp_name] = False
                    return base.Plan(action_plan, maybe_bindings, log)

                # Refinement failed — no valid motion plan bindings found
                if cfg.get('method') in ('zoast', 'VLMB'):
                    write_feedback(exp, "refinement_failed",
                                   prompt_number=cfg["_prompt_number"],
                                   plan=plan_actions,
                                    detail=failure_detail)
                if uses_incremental_blocks:
                    _record_incremental_block(
                        optimistic_plan, failure_info,
                        incremental_blocked_pairs, exp_name,
                    )

                constraints.append(
                    constraint_cls(
                        pddl=new_pddl,
                        task_plan=optimistic_plan,
                        action_plan=action_plan,
                        stream_plan=stream_plan,
                        objects=objects,
                        visited_bindings=visited_bindings,
                        world=world,
                    )
                )
                plan_queue.put((constraint_map[rep_constraints(constraints)], -len(constraints), -timer.toc("plan"), optimistic_plan, new_pddl, constraints[:]))
            failed_iteration_count += 1
            # print("We are not reaching here MIRAS")
        print("[improved.plan] Failure: unable to solve motion plan")
        self._last_problem_failed_by_exp[exp_name] = True
        return base.Plan(action_plan, maybe_bindings, log)

    @classmethod
    def process_streams(
        cls,
        stream_plan: List[StreamInstance],
        optimistic_plan: Sequence[symbolic.PlannerNode],
        action_plan: Sequence[ActionCall],
        timeout: float,
        world: Any,
        bindings: Mapping[str, Object],
        pddl: symbolic.Pddl,
        log: Log,
        cache: Any
    ) -> Tuple[
        Optional[Union[Mapping[str, Object], StreamConstraint]],
        List[PrioritizedBindings],
        Optional[Dict[str, str]],
    ]:
        """Adaptive Algorithm."""
        bindings_queue: queue.PriorityQueue[PrioritizedBindings] = queue.PriorityQueue()
        # If we want to cache, then we just have to look at the task plan and
        # get the current list of bindings
        if cache and optimistic_plan in cache:
            # Add the bindings to the queue
            for binding in cache[optimistic_plan]:
                bindings_queue.put(binding)
        else:
            bindings_queue.put(
                PrioritizedBindings.create(
                    bindings=bindings,
                    stream_plan=stream_plan,
                    idx_plan=0,
                    time_since_start=0.0,
                )
            )

        visited_bindings = []
        blocked_actions = []
        blocked_dict=[]
        current_plan=([str(node.action) for node in optimistic_plan[:]])
        step_by_action_id = {id(action): i + 1 for i, action in enumerate(action_plan)}
        latest_failure_info: Optional[Dict[str, str]] = None
        # print("current plan MIRAS", (current_plan))
        timer = utils.Timer()
        timer.tic("process_streams")
        # print("show me binding queues MIRAS:", bindings_queue.get())

        while not bindings_queue.empty():
            prioritized_binding = bindings_queue.get()
            # print("[mu] prioritized binding", prioritized_binding)

            # breakpoint()
            stream_instance = stream_plan[prioritized_binding.idx_plan]
            if stream_instance.count > 0 and timer.toc("process_streams") > timeout:
                timer_out = timer.toc("process_streams") > timeout
                print(
                    f"[improved.plan] Timeout exceeded {timer_out} and ran out of streams {stream_instance.count > 0}! Breaking on next failure!"
                )
                break
            visited_bindings.append(prioritized_binding)# visited bindings
            # Sample stream.
            # print("[improved.plan] Sampling stream", stream_instance)
            with log.profile_stream(stream_instance):
                # print(f"Instance: {stream_instance}")
                # FAILS HERE
                maybe_bindings = stream_instance.next(prioritized_binding.bindings)
            log.log(stream_instance, maybe_bindings)

            # if isinstance(maybe_bindings, StreamConstraint):
            #    return maybe_bindings, visited_bindings
            # current_plan.append(stream_instance.action_call)
            if maybe_bindings is not None and not isinstance(
                maybe_bindings, StreamConstraint
            ):
                new_bindings = maybe_bindings

                # print(
                #     "[improved.plan] Sample Success:",
                #     stream_instance,
                #     "Action:",
                #     stream_instance.action_call,
                # )
                if prioritized_binding.idx_plan >= len(stream_plan) - 1:
                    # Last stream successfully sampled.
                    # print("here is the last successful sampled stream MIRAS")
                    return new_bindings, visited_bindings, None

                # Queue next stream in stream plan.
                bindings_queue.put(
                    PrioritizedBindings.create(
                        bindings=new_bindings,
                        stream_plan=stream_plan,
                        idx_plan=prioritized_binding.idx_plan + 1,
                        time_since_start=timer.toc("process_streams"),
                    )
                )
            else:

                print(
                    "[improved.plan] Sample Failed:",
                    stream_instance,
                    "Action:",
                    stream_instance.action_call,
                )
                task_step = step_by_action_id.get(id(stream_instance.action_call))
                task_action = (
                    optimistic_plan[task_step - 1].action
                    if task_step is not None and 1 <= task_step <= len(optimistic_plan)
                    else "unknown"
                )
                latest_failure_info = {
                    "stream": str(stream_instance),
                    "task_step": str(task_step) if task_step is not None else "unknown",
                    "task_action": str(task_action),
                    "action_call": str(stream_instance.action_call),
                }


                sstr = (stream_instance)


                # breakpoint()
                if stream_instance.stream.resample(False):
                    # print("we will enter here MIRAS 22")
                    bindings_queue.put(
                        PrioritizedBindings.create(
                            bindings=prioritized_binding.bindings,
                            stream_plan=stream_plan,
                            idx_plan=prioritized_binding.idx_plan,
                            time_since_start=timer.toc("process_streams"),
                        )
                    )


            if stream_instance.stream.resample(True) or stream_instance.count == 0:
                bindings_queue.put(
                    PrioritizedBindings.create(
                        bindings=prioritized_binding.bindings,
                        stream_plan=stream_plan,
                        idx_plan=prioritized_binding.idx_plan,
                        time_since_start=timer.toc("process_streams"),
                    )
                )
        if bindings_queue.empty():
            print("Bindings finished")
        # save current task plan and current successful bindings
        if cache:
            cache[optimistic_plan] = visited_bindings[:-1]
        if isinstance(maybe_bindings, StreamConstraint):
            # print("is if isinstance ??? MIRAS")
            return maybe_bindings, visited_bindings, latest_failure_info
        # breakpoint()
        return None, visited_bindings, latest_failure_info
