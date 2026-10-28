from typing import List, Optional, NamedTuple, Set
import sys
import argparse

sys.path.append('.')
import os
import json
import traceback
import subprocess
import pddlboat as pd
import symbolic
import pybullet_tools as pbt
from coast.algorithms.base import Log

import time


class ConstraintError(Exception):
    """Raised when the VLM-generated constraint function has a bug."""
    pass


def write_feedback(exp: str, failure_type: str, prompt_number: int = 0, **details):
    """Append a feedback entry to the current prompt's feedback list.

    Each prompt cycle gets its own file: feedback_0.json, feedback_1.json, etc.
    Each file contains a JSON list of feedback dicts, one per planning iteration.

    failure_type: "refinement_failed" | "no_plan_found" | "constraint_bug"
    details: additional key-value pairs depending on the failure type.
    """
    entry = {"failure_type": failure_type, **details}
    path = f"envs/{exp}/prompting/output/feedback_{prompt_number}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing list or start fresh
    if os.path.exists(path):
        with open(path, "r") as f:
            feedback_list = json.load(f)
    else:
        feedback_list = []

    feedback_list.append(entry)
    with open(path, "w") as f:
        json.dump(feedback_list, f, indent=2)
    print(f"[feedback] Appended {failure_type} feedback to {path} (now {len(feedback_list)} entries)")


def _run_vrm_in_subprocess(
    exp: str,
    provider: str,
    iteration: int = 0,
    feedback: bool = False,
    model: str = None,
    reasoning_effort: str = None,
    reprompt_failure_threshold: int = 1,
    prompt_number: int = 0,
):
    """Run VLM prompting in a separate subprocess to avoid segfaults.

    PyBullet's GUI mode runs a native OpenGL rendering thread (Intel Mesa)
    that can crash and deliver SIGSEGV to the whole process.  We:
      1. Pause the renderer so the OpenGL thread goes idle.
      2. Spawn a clean subprocess for the VLM HTTPS call.
      3. Re-enable rendering after the subprocess exits.

    stdout/stderr are inherited so output streams in real time.
    """
    model_arg = f', model="{model}"' if model else ""
    reasoning_arg = f", reasoning_effort={reasoning_effort!r}" if reasoning_effort is not None else ""
    feedback_arg = "True" if feedback else "False"
    code = (
        "from coast.vrm import run_vrm; "
        f'run_vrm("{exp}", "{provider}", iteration={iteration}, feedback={feedback_arg}'
        f", reprompt_failure_threshold={reprompt_failure_threshold}"
        f", prompt_number={prompt_number}"
        f"{model_arg}{reasoning_arg})"
    )

    # Pause PyBullet's OpenGL rendering thread so it can't crash while we wait
    pbt.utils.set_renderer(enable=False)
    try:
        # inherit stdout/stderr so prints appear in real time (no buffering)
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", code],   # -u = unbuffered
            cwd=os.getcwd(),
        )
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"VLM subprocess exited with code {proc.returncode}"
            )
    finally:
        # Always re-enable rendering, even if the subprocess failed
        pbt.utils.set_renderer(enable=True)

class PlannerNode(NamedTuple):
    action: str
    state: Set[str]

# prompt the vlm to generate constraints, and return the name
# that should be executed to apply the constraints
def prompt_vlm(
        prompt_strategy,
        failed_iteration_count: int,
        exp,
        provider: str,
        iteration: int = 0,
        feedback: bool = False,
        model: str = None,
        reasoning_effort: str = None,
        reprompt_failure_threshold: int = 1,
        prompt_number: int = 0,
):
    output_dir = f"envs/{exp}/prompting/output"
    # Oracles live outside output/ so they survive the stale-output wipe and stay tracked in git.
    oracle_file = f"envs/{exp}/prompting/oracles/generated_constraints_oracle.py"
    print(f"[fastdownward.prompt_vlm] VLM provider: {provider}, model: {model or '(default)'}, reasoning_effort: {reasoning_effort or '(default)'}")

    latest_constraints = f"{output_dir}/constraints_{prompt_number}.py"

    did_reprompt = False

    if prompt_strategy == 'no constraints':
        print('[fastdownward.prompt_vlm] not using constraints')
        return None, did_reprompt, prompt_number

    elif prompt_strategy == 'on failure':
        target_prompt_number = prompt_number

        if not os.path.exists(latest_constraints):
            # No constraints file yet (first experiment, first iteration) — prompt
            print('[fastdownward.prompt_vlm] no existing constraints, prompting VLM')
            _run_vrm_in_subprocess(
                exp, provider,
                iteration=iteration, feedback=feedback,
                model=model,
                reasoning_effort=reasoning_effort,
                reprompt_failure_threshold=reprompt_failure_threshold,
                prompt_number=target_prompt_number,
            )
            did_reprompt = True

        elif failed_iteration_count >= reprompt_failure_threshold:
            # Re-prompt only after enough failed iterations.
            target_prompt_number = prompt_number + 1
            print(
                "[fastdownward.prompt_vlm] reprompting VLM due to failures: "
                f"{failed_iteration_count} >= {reprompt_failure_threshold}"
            )
            _run_vrm_in_subprocess(
                exp, provider,
                iteration=iteration, feedback=feedback,
                model=model,
                reasoning_effort=reasoning_effort,
                reprompt_failure_threshold=reprompt_failure_threshold,
                prompt_number=target_prompt_number,
            )
            did_reprompt = True

        else:
            print(
                "[fastdownward.prompt_vlm] reusing existing constraints "
                f"(failed_iterations={failed_iteration_count}, "
                f"threshold={reprompt_failure_threshold})"
            )
        active_constraints = f"{output_dir}/constraints_{target_prompt_number}.py"
        return active_constraints, did_reprompt, target_prompt_number

    elif prompt_strategy == 'oracle':
        print('[fastdownward.prompt_vlm] not prompting vlm, using working oracle code')
        return oracle_file, did_reprompt, prompt_number



    else:
        raise ValueError(f"Invalid prompt strategy: {prompt_strategy}")


def apply_constraints(file_name, problem, domain, planner, pd):
    # "" and "None" arrive from the subprocess CLI, which can only pass strings;
    # they mean the same thing as None ("no constraints for this method").
    if file_name is None or file_name in ("", "None"):
        return
    with open(file_name, "r") as f:
        code = f.read()
    try:
        exec(code)
    except Exception as e:
        raise ConstraintError(
            f"Error executing constraints from {file_name}: {e}\n"
            f"{traceback.format_exc()}"
        ) from e


# SAT-planner backend for the Z3-based methods (zoast, incremental, coast_z3).
# These methods drive a SAT/SMT planner through pddlboat's shared Planner
# interface, so the underlying encoding is swappable without touching the
# constraint/solve/blocking code. Set the config's "z3_encoding" to
# "madagascar" (default) or "satplan" to choose the backend.
_SAT_PLANNER_BACKENDS = {
    "satplan": pd.Z3Planner,
    "madagascar": pd.MadagascarPlanner,
}


def make_sat_planner(problem, horizon_max=0, encoding="madagascar", semantics=None):
    """Build the SAT-based planner for the Z3-family methods.

    Defaults to MadagascarPlanner; pass encoding="satplan" to swap in
    Z3Planner. Both expose the same Planner interface, so callers are
    unchanged. Only the horizon (shared by both backends) is configured here.
    """
    backend = str(encoding).lower()
    planner_cls = _SAT_PLANNER_BACKENDS[backend]
    opts = planner_cls.Options()

    # opts.diagnostics = True
    # opts.dump_clauses = True

    if horizon_max > 0:
        opts.horizon_max = horizon_max
    # Step semantics: the default EXISTS_STEP packs several actions into one
    # planning step and only requires that *some* serialisation of them be
    # valid, so a state that exists only between two actions of the same step
    # is invisible to constraints and invariants.  FORALL_STEP requires every
    # serialisation to be valid, which stops interfering actions sharing a
    # step.  Domains that state rules as "this situation must never exist"
    # need it; set "sat_semantics": "forall_step" in the config.
    if semantics is not None:
        try:
            opts.semantics = getattr(planner_cls.Semantics, str(semantics).upper())
        except AttributeError as exc:
            raise ValueError(
                f"unknown sat_semantics {semantics!r}; expected 'exists_step' or 'forall_step'"
            ) from exc
    print(f"[fastdownward.make_sat_planner] backend = {backend}")
    for f in ("horizon_max", "horizon_min", "parallel", "semantics", "diagnostics", "dump_clauses"):
        if hasattr(opts, f):  # Z3Planner.Options has no `semantics`
            print(f, "=", getattr(opts, f))
    
    return planner_cls(problem, opts)


def task_plan(
     cfg,
     failed_iteration_count: int,
     problem_state: str,
     pddl: symbolic.Pddl,
     log: Log = None,
     iteration: int = 0,
) -> Optional[List[symbolic.PlannerNode]]:

    # RESOLVE PATHS -----------------------------------------
    exp = cfg['exp']
    method = cfg['method']
    prompt_strategy = cfg['prompt_strategy']
    provider = cfg['provider']
    feedback_raw = cfg.get("feedback", False)
    if isinstance(feedback_raw, str):
        feedback = feedback_raw.strip().lower() in {"1", "true", "yes", "on", "feedback"}
    else:
        feedback = bool(feedback_raw)
    model = cfg.get('model', None)
    reasoning_effort = cfg.get("reasoning_effort", None)
    reprompt_failure_threshold = max(1, int(cfg.get("reprompt_failure_threshold", 1)))
    prompt_number = int(cfg.get("_prompt_number", 0))
    plan_as_sub = cfg['plan_as_subproc']
    timeout = cfg['timeout']
    z3_encoding = str(cfg.get('z3_encoding', 'madagascar')).lower()

    # problem_state may come as a pddl spec, or a filename containing the pddl spec.
    # Environments that build the problem file in run.py (kuka, tetris) have already
    # written it; for others we write it here.
    path_to_problem = f"envs/{exp}/temp_problem.pddl"
    if exp in ('containers', 'packing', 'cooking'):
        print(f"[fastdownward.task_plan] experiment is {exp}, temp_problem already written by run.py")
    else:
        with open(path_to_problem, "w") as problem_file:
            problem_file.write(problem_state)


    # INIT PLANNER -----------------------------------------
    # For zoast: use the timestep-free domain for Z3 planning.
    # The symbolic.Pddl object (pddl) uses the timestep-bearing domain for state tracking.
    _zoast_envs = ('containers', 'packing', 'cooking')
    if method in ('zoast', 'incremental') and exp in _zoast_envs:
        domain = pd.parse_domain(f'envs/{exp}/domain_z3.pddl')
    else:
        domain = pd.parse_domain(pddl.domain_pddl)
    print("[fastdownward.task_plan] done parsing domain from", pddl.domain_pddl)
    problem = pd.parse_problem(domain, path_to_problem)
    print("[fastdownward.task_plan] done parsing problem from", path_to_problem)

    # For incremental method: restrict the horizon so iterative deepening
    # can be driven from the outer planning loop.
    horizon_max = 0
    if method == 'incremental' and '_incremental_horizon' in cfg:
        horizon_max = int(cfg['_incremental_horizon'])
        print(f"[fastdownward.task_plan] incremental horizon_max = {horizon_max}")

    planner = make_sat_planner(problem, horizon_max=horizon_max, encoding=z3_encoding,
                               semantics=cfg.get("sat_semantics"))
    print("[fastdownward.task_plan] planner initialized")

    

    # APPLY CONSTRAINTS -----------------------------------------
    constraints_file = ""
    if method == 'zoast':
        print('[fastdownward.task_plan] prompting...')
        with log.profile("prompting"):
            constraints_file, did_reprompt, updated_prompt_number = prompt_vlm(
                prompt_strategy,
                failed_iteration_count,
                exp,
                provider,
                iteration=iteration,
                feedback=feedback,
                model=model,
                reasoning_effort=reasoning_effort,
                reprompt_failure_threshold=reprompt_failure_threshold,
                prompt_number=prompt_number,
            )
            cfg["_prompt_number"] = updated_prompt_number
            # Let outer planning loop reset failure counter after a re-prompt.
            cfg["_reprompted_this_iteration"] = did_reprompt
        # If no actual VLM call was made, zero out the recorded prompting time
        # so the CSV doesn't show spurious overhead as "prompting".
        if not did_reprompt:
            log._tictocs["prompting"][-1] = 0.0
        else:
            log.num_prompts += 1
        apply_constraints(constraints_file, problem, domain, planner, pd)

    elif method == 'incremental':
        # Pure incremental: temp_constraints is the only constraint source.
        pass  # handled below by the shared incremental-blocks loader

    # Apply incremental state-action blocks (temp_constraints.py) for any
    # Z3-based method.  For 'zoast' this gives plan diversity between
    # reprompts; for 'incremental' it is the primary constraint mechanism.
    if method in ('zoast', 'incremental'):
        temp_constraints_file = f"envs/{exp}/temp_constraints.py"
        if os.path.exists(temp_constraints_file):
            print(f'[fastdownward.task_plan] applying incremental blocks from {temp_constraints_file}')
            apply_constraints(temp_constraints_file, problem, domain, planner, pd)
        else:
            print('[fastdownward.task_plan] no incremental blocks yet')




    # ACTUALLY CALL THE PLANNER --------------------------------
    if plan_as_sub:
            path_to_plan = f"envs/{exp}/temp_plan.txt"
            # Resolve the domain file for the subprocess.
            # For coast: use pddl.domain_pddl which may be temp_domain.pddl
            # (constraint-modified). For zoast/incremental with kuka/tetris:
            # use domain_z3 (timestep-free). For other combos pddl.domain_pddl
            # already points to domain_z3.
            if method in ('zoast', 'incremental') and exp in ('containers', 'packing', 'cooking'):
                sub_domain = f"envs/{exp}/domain_z3.pddl"
            else:
                sub_domain = pddl.domain_pddl
            # Build the subprocess command with both VLM constraints and
            # incremental state-action blocks (if they exist).
            sub_cmd = [
                "python", "coast/fastdownward.py",
                sub_domain,
                f"{path_to_problem}",
                f"envs/{exp}/temp_plan.txt",
                # "" (not "None") when there is no constraints file: argparse
                # would otherwise hand the child the literal string "None" and
                # it would die trying to open a file by that name.
                constraints_file if constraints_file else "",
            ]
            temp_constraints_file = f"envs/{exp}/temp_constraints.py"
            if method in ('zoast', 'incremental') and os.path.exists(temp_constraints_file):
                sub_cmd.append(temp_constraints_file)
            # Pass the incremental horizon to the subprocess so it can limit
            # the Z3 solver depth for iterative deepening.
            if method == 'incremental' and '_incremental_horizon' in cfg:
                sub_cmd.extend(["--horizon-max", str(int(cfg['_incremental_horizon']))])
            sub_cmd.extend(["--z3-encoding", z3_encoding])
            # Remove any plan left by a previous iteration. Without this, a
            # subprocess that crashes (segfault, UNSAT, bad constraints) leaves
            # the parent reading a stale temp_plan.txt and silently attributing
            # an older problem's plan to this one.
            if os.path.exists(path_to_plan):
                os.remove(path_to_plan)
            with log.profile("planning"):
                try:
                    completed = subprocess.run(sub_cmd, timeout=timeout)
                except subprocess.TimeoutExpired:
                    print('[fastdownward.task_plan] subprocess timed out')
                    return None
            if completed.returncode != 0 or not os.path.exists(path_to_plan):
                print(
                    f'[fastdownward.task_plan] subprocess produced no plan '
                    f'(returncode {completed.returncode})'
                )
                return None

            state = pddl.initial_state
            task_plan = [PlannerNode("", state)]
            with open(path_to_plan) as f:
                task_plan_str = f.read()
            steps = task_plan_str.split("\n")[:-1]
            for a in steps:
                state = pddl.next_state(state, a)
                task_plan.append(PlannerNode(a, state))
            return task_plan

    else:
        print('[fastdownward.task_plan] planning with Z3...')
        with log.profile("planning"):
            #TODO: Replace with option to run as subproccess in which fast downwawrd is called through command subprocess
            # passing necessary files as args
            plan = pd.Plan(problem)
            solved = planner.solve(plan)

        if not solved:
            print('[fastdownward.task_plan] Z3 planner found no valid plan (constraints may be too restrictive)')
            del planner
            del problem
            del domain
            return None

        steps = plan.get_steps()
        print("[fastdownward.task_plan] steps", steps)

        state = pddl.initial_state
        task_plan = [PlannerNode("", state)]

        # Keep Z3 action signatures unchanged (no synthetic timestep parameters).
        for i, a in enumerate(steps):
            all_symbols = str(a).strip()[1:-1].split(" ")
            a = f"{all_symbols[0]}({','.join(all_symbols[1:])})"
        
            state = pddl.next_state(state, a)
            task_plan.append(PlannerNode(a, state))
        
        print('[fastdownward.task_plan] deleting planner')
        start_time = time.time()
        del planner
        del problem
        del domain
        print('[fastdownward.task_plan] task planning done===========================')
        return task_plan

def task_plan_subprocess(
    domain_pddl: str,
    problem_pddl: str,
    output_file: str,
    constraints_file: str,
    incremental_constraints_file: str = "",
    horizon_max: int = 0,
    encoding: str = "madagascar",
    ) -> Optional[List[symbolic.PlannerNode]]:

    print('[fastdownward.task_plan_subprocess] SUBPROCESS STARTED')

    domain = pd.parse_domain(domain_pddl)
    print("[fastdownward.task_plan_subprocess] done parsing domain from", domain_pddl)
    problem = pd.parse_problem(domain, problem_pddl)
    print("[fastdownward.task_plan_subprocess] done parsing problem from", problem_pddl)


    print('[fastdownward.task_plan_subprocess] planning with SAT planner...')
    planner = make_sat_planner(problem, horizon_max=horizon_max, encoding=encoding)
    if constraints_file != "":      # If constraint file is given, aka if method is ZOAST
        print("[fastdownward.task_plan_subprocess] APPLYING CONSTRAINTS - File: ", constraints_file)
        apply_constraints(constraints_file, problem, domain, planner, pd)
    if incremental_constraints_file != "" and os.path.exists(incremental_constraints_file):
        print("[fastdownward.task_plan_subprocess] APPLYING INCREMENTAL BLOCKS - File: ", incremental_constraints_file)
        apply_constraints(incremental_constraints_file, problem, domain, planner, pd)
    plan_start_time = time.time()
    plan = pd.Plan(problem)
    solved = planner.solve(plan)

    plan_end_time = time.time()
    planning_time = plan_end_time - plan_start_time
    print(f"[fastdownward.task_plan_subprocess] Planning time: {planning_time} seconds")

    if not solved:
        # Exit non-zero *without* writing the output file, so the parent's
        # returncode check turns this into "no plan found" rather than an empty
        # plan. Iterative deepening ('incremental') depends on being able to
        # tell UNSAT from a zero-action plan.
        print('[fastdownward.task_plan_subprocess] no valid plan found')
        exit(2)

    steps = plan.get_steps()
    print("[fastdownward.task_plan_subprocess] steps", steps)

    task_plan_str = []
    for a in steps:
        all_symbols = str(a).strip()[1:-1].split(" ")
        a = f"{all_symbols[0]}({','.join(all_symbols[1:])})"
        task_plan_str.append(a)


    buffer = "" # f"{planning_time}\n"  # FIRST LINE IS THE PLANNING TIME
    for step in task_plan_str:
        buffer += f"{step}\n"
    # buffer += "DONE\n"
    with open(output_file, 'w') as f:
        f.write(buffer)
    
    del planner
    del problem
    del domain

    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task planning subprocess')
    parser.add_argument('domain_pddl', help='Path to domain PDDL file')
    parser.add_argument('problem_pddl', help='Path to problem PDDL file')
    parser.add_argument('output_file', help='Path to output file')
    parser.add_argument('constraints_file', help="Path to constraints file")
    parser.add_argument('incremental_constraints_file', nargs='?', default='',
                        help="Path to incremental state-action blocks file (optional)")
    parser.add_argument('--horizon-max', type=int, default=0,
                        help="Z3 solver horizon_max for iterative deepening (0 = unlimited)")
    parser.add_argument('--z3-encoding', default='madagascar', choices=['satplan', 'madagascar'],
                        help="SAT planner backend: 'satplan' (Z3Planner) or 'madagascar' (default)")
    args = parser.parse_args()

    task_plan_subprocess(
        domain_pddl=args.domain_pddl,
        problem_pddl=args.problem_pddl,
        output_file=args.output_file,
        constraints_file=args.constraints_file,
        incremental_constraints_file=args.incremental_constraints_file,
        horizon_max=args.horizon_max,
        encoding=args.z3_encoding,
    )