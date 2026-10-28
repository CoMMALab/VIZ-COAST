"""VLM Baseline (VLMB) task planner.

Prompts a Vision-Language Model to directly produce a PDDL action sequence,
without Z3 or Fast Downward.  Uses multi-turn prompting with structured
feedback from the planning loop (same feedback files as zoast).
"""

from typing import List, Optional
import os
import json
import base64
import textwrap

import symbolic
from coast.algorithms.base import Log
from coast.vlm import get_vlm_client
from coast.fastdownward import PlannerNode
from coast.vrm import _format_feedback_list


def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _grasp_info(exp: str) -> str:
    """Return a human-readable grasp specification for the environment."""
    if exp == "shelf":
        return "grasp the SIDE of the object, approaching grid from negative Y direction, end-effector at fixed height throughout motion"
    elif exp == "rack":
        return (
            "grasp the TOP of the object, hover directly above the target position and "
            "descend straight down along z; the object never moves sideways while it is "
            "near the rack"
        )
    elif exp in ("containers", "packing", "cooking"):
        return "grasp the TOP of the object"
    return "grasp the TOP of the object (default)"


def _build_initial_prompt(
    exp: str,
    domain_str: str,
    problem_str: str,
    scene_description: str,
) -> str:
    """Build the initial VLMB prompt."""
    grasp = _grasp_info(exp)
    return textwrap.dedent(f"""\
I am performing task-and-motion planning in this environment.
My robot arm grasps objects according to this specification: {grasp}
The textual scene description:
{scene_description}
The PDDL domain:
{domain_str}
The PDDL problem:
{problem_str}
Given the domain and problem, create a plan using actions contained in the PDDL domain \
to transition the initial state into the goal state.
Keep in mind that the PDDL domain is an abstraction and valid task \
plans may yet fail grounding to motion plans.
Your response should ONLY include PDDL actions, one per line, with no extra text.""")


def _print_thinking(cfg, client) -> None:
    """Print the VLM's chain-of-thought/reasoning trace to the terminal when
    the --show-thinking flag is set. Not all providers/models emit one (e.g.
    reasoning disabled, or the model doesn't support summaries)."""
    if not cfg.get("show_thinking", False):
        return
    reasoning = getattr(client, "last_reasoning", None)
    print("\n[vlmb.task_plan] ===== Chain of thought =====")
    print(reasoning if reasoning else "(no reasoning trace returned by the provider/model)")
    print("[vlmb.task_plan] ===== End chain of thought =====\n")


def _build_feedback_prompt(feedback_list: list, problem_str: str) -> str:
    """Build a feedback prompt from the structured feedback entries."""
    feedback_summary = _format_feedback_list(feedback_list)
    return textwrap.dedent(f"""\
Your previous plan failed. Here is what happened:
{feedback_summary}

The PDDL problem instance is:
{problem_str}

Generate a new, different plan that avoids the above failures.
Your response should ONLY include PDDL actions, one per line, with no extra text.""")


def task_plan(
    cfg,
    failed_iteration_count: int,
    problem_state: str,
    pddl: symbolic.Pddl,
    log: Log = None,
    iteration: int = 0,
) -> Optional[List[PlannerNode]]:
    """Generate a task plan by prompting a VLM with domain, problem, and scene image."""

    print("[vlmb.task_plan] Running VLM baseline...\n")
    exp = cfg['exp']
    provider = cfg['provider']
    model = cfg.get('model', None)
    reasoning_effort = cfg.get("reasoning_effort", None)
    prompt_number = int(cfg.get("_prompt_number", 0))

    # Ensure the problem file is written.
    path_to_problem = f"envs/{exp}/temp_problem.pddl"
    if exp in ('containers', 'packing', 'cooking'):
        print(f"[vlmb.task_plan] experiment is {exp}, temp_problem already written by run.py")
    else:
        with open(path_to_problem, "w") as f:
            f.write(problem_state)

    # Paths
    output_dir = f"envs/{exp}/prompting/output"
    input_dir = f"envs/{exp}/prompting/input"
    os.makedirs(output_dir, exist_ok=True)

    # SETUP ==================================
    # pddl.domain_pddl is the exact file used for planning (run.py generates
    # a size-matched *_generated.pddl sibling when the domain template
    # declares no block/location constants).
    domain_str = _read_file(pddl.domain_pddl)
    problem_str = _read_file(path_to_problem)
    scene_desc_path = os.path.join(input_dir, "scene_description.txt")
    scene_description = _read_file(scene_desc_path) if os.path.exists(scene_desc_path) else ""

    with open(os.path.join(input_dir, "default_camera_capture.png"), "rb") as f:
        scene_image = base64.b64encode(f.read()).decode("utf-8")

    client_kwargs = {}
    if model:
        client_kwargs["model"] = model
    if reasoning_effort is not None:
        client_kwargs["reasoning_effort"] = reasoning_effort
    client = get_vlm_client(provider, **client_kwargs)

    # PROMPTING ==================================
    # VLMB always reprompts after failure (threshold = 1).
    initial_prompt = _build_initial_prompt(exp, domain_str, problem_str, scene_description)

    # Save the initial prompt for multi-turn replay.
    initial_prompt_path = os.path.join(output_dir, "VLMB_initial_prompt.txt")
    if prompt_number == 0:
        with open(initial_prompt_path, "w", encoding="utf-8") as f:
            f.write(initial_prompt)
    else:
        # Re-read saved prompt so the conversation stays consistent.
        initial_prompt = _read_file(initial_prompt_path)

    target_prompt_number = prompt_number
    need_reprompt = False

    plan_path = os.path.join(output_dir, f"VLMB_plan_{prompt_number}.txt")

    if not os.path.exists(plan_path):
        # First prompt or plan file missing.
        need_reprompt = True
    elif failed_iteration_count >= 1:
        # Plan failed; advance prompt number and reprompt with feedback.
        target_prompt_number = prompt_number + 1
        plan_path = os.path.join(output_dir, f"VLMB_plan_{target_prompt_number}.txt")
        need_reprompt = True

    if need_reprompt:
        with log.profile("prompting"):
            if target_prompt_number == 0:
                # First prompt: single-turn.
                print("[vlmb.task_plan] Initial prompt (single-turn)")
                plan_text = client.generate(
                    text=initial_prompt,
                    images=[scene_image],
                )
            else:
                # Feedback reprompt: multi-turn conversation.
                print(f"[vlmb.task_plan] Feedback reprompt (prompt_number={target_prompt_number})")
                messages = [{"role": "user", "text": initial_prompt}]

                for i in range(target_prompt_number):
                    # Previous plan (assistant response).
                    prev_plan_path = os.path.join(output_dir, f"VLMB_plan_{i}.txt")
                    prev_plan = _read_file(prev_plan_path)
                    messages.append({"role": "assistant", "text": prev_plan})

                    # Feedback (user turn).
                    feedback_path = os.path.join(output_dir, f"feedback_{i}.json")
                    if os.path.exists(feedback_path):
                        with open(feedback_path, "r") as f:
                            feedback_list = json.load(f)
                        feedback_text = _build_feedback_prompt(feedback_list, problem_str)
                    else:
                        feedback_text = (
                            "The previous plan failed during motion planning. "
                            "Generate a new, different plan. "
                            "Your response should ONLY include PDDL actions, one per line."
                        )
                    messages.append({"role": "user", "text": feedback_text})

                print(f"[vlmb.task_plan] Conversation has {len(messages)} turns")
                plan_text = client.generate_multiturn(
                    messages=messages,
                    images=[scene_image],
                )

        _print_thinking(cfg, client)

        # Save the plan.
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan_text)
        log.num_prompts += 1
        cfg["_prompt_number"] = target_prompt_number
        cfg["_reprompted_this_iteration"] = True
        print(f"[vlmb.task_plan] Saved plan to {plan_path}")
    else:
        # Reuse existing plan (edge case: shouldn't normally happen with threshold=1).
        plan_text = _read_file(plan_path)
        cfg["_reprompted_this_iteration"] = False
        print(f"[vlmb.task_plan] Reusing plan from {plan_path}")

    # Also write to legacy path for compatibility.
    with open(f"envs/{exp}/VLMB_plan.txt", "w", encoding="utf-8") as f:
        f.write(plan_text)

    # PARSE PLAN ==================================
    actions = plan_text.strip().split("\n")
    print(f"[vlmb.task_plan] Length of actions: {len(actions)}")
    state = pddl.initial_state
    task_plan = [PlannerNode("", state)]
    for a in actions:
        a = str(a).strip()
        if not a:
            continue
        # Accept both "(pick b1 loc5)" and "pick b1 loc5".
        if a.startswith("(") and a.endswith(")"):
            a = a[1:-1]  # strip outer parens
        # Skip lines that don't look like actions (e.g. markdown, commentary).
        parts = a.split()
        if not parts or parts[0].startswith("#") or parts[0].startswith("```"):
            continue
        print(f"({a})")
        a_fmt = f"{parts[0]}({','.join(parts[1:])})"
        state = pddl.next_state(state, a_fmt)
        task_plan.append(PlannerNode(a_fmt, state))
    print(f"[vlmb.task_plan] Parsed {len(task_plan) - 1} action(s)")
    return task_plan
