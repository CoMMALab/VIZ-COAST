from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Any,
    Tuple,
)

import symbolic  # type: ignore
import pddlboat as pd
from coast.fastdownward import task_plan
from coast.object import StreamObject, Object
from coast.stream import (
    CertifiedFact,
    CertifiedObject,
    Stream,
    StreamInstance,
    StreamState,
)

from coast import fastdownward, fdplan, vlmb, utils
from coast.algorithms.base import Log

# All method names that use coast-style PDDL constraint modification.
COAST_METHODS = ('coast_fd', 'coast_z3')
NON_COAST_METHODS = ('zoast', 'VLMB', 'incremental')

import time
import os
import subprocess
from coast.fastdownward import PlannerNode


def create_problem(
    name: str,
    pddl: symbolic.Pddl,
    stream_state: Set[str],
) -> symbolic.Problem:
    constants = set(obj.name for obj in pddl.constants)
    problem = symbolic.Problem(name, domain=pddl.name)
    """for obj in stream_state.pddl_objects():
        if obj.name in constants:
            continue
        problem.add_object(obj.name, object_type=obj.type)
    """
    problem.set_initial_state(stream_state)
    problem.set_goal(str(pddl.goal))
    return problem


def preimage(
    task_plan: Sequence[symbolic.PlannerNode], pddl: symbolic.Pddl
) -> Tuple[Set[str], Set[str]]:
    def get_action_propositions(pddl: symbolic.Pddl, action: str) -> Set[str]:
        pre_dnf, _ = symbolic.DisjunctiveFormula.normalize_conditions(pddl, action)
        props = set()
        for conj in pre_dnf.conjunctions:
            props.update(conj.pos)
            props.update(conj.neg)
        return props

    def get_proposition_objects(propositions: Set[str]) -> Set[str]:
        objects = set()
        for prop in propositions:
            objects.update(symbolic.parse_args(prop))
        return objects

    propositions = set()
    objects = set()
    for action_state in task_plan[1:]:
        action_props = get_action_propositions(pddl, action_state.action)
        action_objects = get_proposition_objects(action_props)
        propositions.update(action_props)
        objects.update(action_objects)

    return propositions, objects


def ordered_preimage(
    task_plan: Sequence[symbolic.PlannerNode], pddl: symbolic.Pddl
) -> Tuple[List[str], Set[str]]:
    def get_action_propositions(pddl: symbolic.Pddl, action: str) -> List[str]:
        # print("action for normalizing", action)
        pre_dnf = symbolic.DisjunctiveFormula.normalize_preconditions(pddl, action)
        # print(pre_dnf)
        props = {}
        for conj in pre_dnf.conjunctions:
            for i in list(conj.pos):
                if i not in props:
                    props[i] = None
                else:
                    break
            for i in list(conj.neg):
                if i not in props:
                    props[i] = None
                else:
                    break
        prop_list = []
        for key in props:
            prop_list.append(key)
        return prop_list

    def get_proposition_objects(propositions: List[str]) -> Set[str]:
        objects = set()
        for prop in propositions:
            objects.update(symbolic.parse_args(prop))
        return objects

    propositions: List[str] = []
    objects: Set[str] = set()
    timestep = 1
    for action_state in task_plan[1:]:
        action_props = get_action_propositions(pddl, action_state.action)
        # Add the timestep of the task plan into each action prop
        for i in range(len(action_props)):
            action_props[i] += f"-t{timestep}"
        timestep += 1
        # action_objects = get_proposition_objects(action_props)
        propositions.extend(list(action_props))
        # objects.update(action_objects)
    return propositions, objects


def instantiate(
    streams: Sequence[Stream],
    stream_state: StreamState,
    stream_pddl: symbolic.Pddl,
) -> Sequence[StreamInstance]:
    state = stream_state.pddl_state()
    return [
        stream(inputs, stream_pddl, stream_state)
        for stream in streams
        for inputs in stream.valid_inputs(state, stream_pddl)
    ]


def get_level(
    stream_state: StreamState,
    stream_instance: StreamInstance,
) -> int:
    input_levels = [
        stream_state.propositions[prop].level for prop in stream_instance.domain
    ]
    input_levels += [
        stream_state.objects[obj.name].level for obj in stream_instance.inputs
    ]
    if stream_instance.apply_all:
        return 0

    return 1 + stream_instance.count + max(input_levels)


def add_certified_from_prev(
    stream_state: StreamState,
    stream_instance: StreamInstance,
    prev_state: Dict[str, Object],
    bindings: Dict[Any, Any],
    logging: Dict[str, Tuple[int, int]],
    global_motion_cache: Dict[Tuple[Any], Any],
) -> Optional[Tuple[Dict[str, Any], Dict[Any, Any]]]:
    return stream_instance.next(prev_state, bindings, logging, global_motion_cache)


def add_certified(
    stream_state: StreamState,
    stream_instance: StreamInstance,
    output_fn: Callable[[StreamInstance], Optional[Dict[str, StreamObject]]],
) -> Optional[Iterable[StreamObject]]:
    level = get_level(stream_state, stream_instance)
    output_objects = output_fn(stream_instance)
    if output_objects is None:
        return None

    # Add certified facts.
    for fact in stream_instance.certified:
        if fact in stream_state.propositions:
            continue
        stream_state.propositions[fact] = CertifiedFact(fact, level, stream_instance)

    # Add certified object types.
    for obj in output_objects.values():
        if obj.name in stream_state.objects:
            continue
        stream_state.objects[obj.name] = CertifiedObject(
            StreamObject(obj.name, obj.type, obj.value), level, stream_instance
        )

    return output_objects.values()


def apply_streams(
    stream_pddl: symbolic.Pddl,
    streams: Sequence[Stream],
    stream_state: StreamState,
    level: int,
    output_fn: Callable,
) -> StreamState:
    problem = create_problem("stream", stream_pddl, stream_state)
    stream_pddl = symbolic.Pddl(stream_pddl.domain_pddl, str(problem))

    # The paper suggests calling instantiate() inside the k for loop. But this
    # would require creating a new stream_pddl every time add_certified()
    # is called, since the generated stream instance outputs need to be added to
    # the stream_pddl. This is expensive and doesn't seem to offer any
    # algorithmic advantage, so we instantiate() outside the loop.
    stream_instances = instantiate(streams, stream_state, stream_pddl)
    new_stream_state = stream_state.copy()
    for k in range(0, level + 1):
        for stream_instance in stream_instances:
            if get_level(stream_state, stream_instance) != k:
                continue
            add_certified(new_stream_state, stream_instance, output_fn)
    return new_stream_state


def search(
    cfg: Any,
    failed_iteration_count: int,
    pddl: symbolic.Pddl,
    stream_state: Set[str],
    max_depth: int = 6,
    timeout: float = 0.0,
    verbose: bool = True,
    use_symbolic: bool = False,
    task_horizon: int = 50,
    failed_last_action: List[str] = None,  # prover wto eto znachit
    planner_condition_list: List = None,
    log: Log = None,
    iteration: int = 0,
) -> Tuple[Optional[List[symbolic.PlannerNode]], symbolic.Pddl]:

    if cfg['method'] in COAST_METHODS:
        problem = create_problem(pddl.name, pddl, stream_state)
        pddl = symbolic.Pddl(pddl.domain_pddl, str(problem))
        problem_state = str(problem)
    elif cfg['method'] in NON_COAST_METHODS:
        problem_state = pddl.problem_pddl

    # not used
    if use_symbolic:
        planner = symbolic.Planner(pddl)
        bfs = symbolic.BreadthFirstSearch(
            planner.root, max_depth=max_depth, timeout=timeout, verbose=verbose
        )

        task_plan = None
        for task_plan in bfs:
            break
    else:

        if 'subprocess_search' in cfg.keys() and cfg['subprocess_search'] == True:
            print("[common.search] calling fastdownward.task_plan in a subprocess")
            output_file = 'z3_plan.txt'
            
            # Clean up any existing output file
            if os.path.exists(output_file):
                os.remove(output_file)
            proc = subprocess.Popen(['python3',
                        'coast/fastdownward.py',
                        pddl.domain_pddl,
                        pddl.problem_pddl,
                        output_file],
                    )

            # Wait for the output file to appear
            print(f"Waiting for {output_file} to appear")
            while not os.path.exists(output_file):
                time.sleep(0.1)  # Check every 100ms
            time.sleep(1)

            proc.terminate()
            print('[common.search] task plan subprocess done')
            with open(output_file, 'r') as f:
                lines = f.readlines()
                if lines[-1].strip() != "DONE":
                    raise ValueError(f"Output file {output_file} does not contain DONE")

                planning_time = lines[0].strip()
                task_plan_str = lines[1:-1]
                task_plan_str = [line.strip() for line in task_plan_str]
            
            log.force_set_time("planning", float(planning_time))

            state = pddl.initial_state
            task_plan = [PlannerNode("", state)]
            for line in task_plan_str:
                state = pddl.next_state(state, line)
                task_plan.append(PlannerNode(line, state))
            
        else:
            if cfg['method'] == 'coast_fd':
                # Use original Fast Downward planner with coast-style constraints.
                task_plan = fdplan.task_plan(
                    log=log,
                    exp=cfg['exp'],
                    problem_state=problem_state,
                    pddl=pddl,
                )
            elif cfg['method'] == 'VLMB':
                # VLM baseline: prompt a VLM to produce an action sequence directly.
                task_plan = vlmb.task_plan(
                    cfg=cfg,
                    failed_iteration_count=failed_iteration_count,
                    problem_state=problem_state,
                    pddl=pddl,
                    iteration=iteration,
                    log=log,
                )
            else:
                # All other methods (coast_z3, zoast, incremental, etc.)
                # go through the Z3-based task planner.
                task_plan = fastdownward.task_plan(
                    cfg=cfg,
                    failed_iteration_count=failed_iteration_count,
                    problem_state=problem_state,
                    pddl=pddl,
                    iteration=iteration,
                    log=log,
                )

    # Force collection of C++ objects (Z3 solver, pddlboat) from task_plan
    print('[common.search] force collection of C++ objects')
    import gc
    gc.collect()
    print('[common.search] force collection of C++ objects done')

    return task_plan, pddl
