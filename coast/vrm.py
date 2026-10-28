from logging import FATAL
from time import sleep
import time
import json
import base64
import os
from coast.vlm import get_vlm_client
from coast.utils import resolve_generated_pddl
import re
import textwrap

_SERVER_RETRY_WAIT = 10  # seconds to wait before retrying on server overload


def _is_server_overload(exc: Exception) -> bool:
    """Return True if the exception indicates a transient server overload."""
    err_str = str(exc).lower()
    type_name = type(exc).__name__.lower()
    keywords = (
        "503", "overloaded", "unavailable", "429", "rate_limit", "rate limit",
        "resource_exhausted", "resource exhausted", "high demand", "servererror",
    )
    return any(kw in err_str or kw in type_name for kw in keywords)


REMINDERS = """
Implementation Rules:
1. Declare all required variables inside the function.
2. Use only the allowed objects, predicates and actions defined below. Do not create new objects or dictionaries with objects.
3. Assess all blocked actions in relation to the provided constraints.
4. Bind correct predicate arguments (objects and locations).
5. Use valid and logically correct conditional logic.
6. Ensure the code is fully complete, readable, and maintains logical correctness.
7. Constraints may involve negated conditions using pd.make_not.
8. Do NOT change the function signature
9. Do NOT use code fences

Reminders:
- You must not define new predicates or actions beyond those listed.
- Every action you block must be associated with a valid predicate condition.
- Objects and locations may move or change state based on actions, so constraints must reflect logical dependencies.
- Always respect the order of parameters in action definitions.
- Do not provide examples.
- Your output should ONLY include the text of the function definition.
"""

def _read_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        return f.read()

def _read_feedback(exp: str):
    """Read the feedback JSON written by the planner after the previous iteration.

    Returns None if the file doesn't exist (e.g. first iteration).
    """
    path = f"envs/{exp}/prompting/output/feedback.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _format_feedback(feedback: dict) -> str:
    """Turn a feedback dict into a human-readable summary for the VLM prompt."""
    ft = feedback.get("failure_type", "unknown")

    if ft == "refinement_failed":
        plan = feedback.get("plan", [])
        detail = feedback.get("detail", "")
        plan_str = " → ".join(plan) if plan else "(empty)"
        return (
            f"The generated constraints produced a task plan [{plan_str}], "
            f"but it failed during downward refinement to motion plans. "
            f"Failure diagnostics: {detail}."
            f"You may have forgotten to consider some failure modes, or incorrectly"
            f"encoded some constraints."
        )

    elif ft == "no_plan_found":
        return (
            f"The Z3 planner could not find any valid task plan satisfying both the domain"
            f"rules and the constraints. Downward refinement could not be attempted."
            f"The constraints may be too restrictive."
        )

    elif ft == "constraint_bug":
        error = feedback.get("error", "(unknown error)")
        return (
            f"The generated constraint function raised an error when executed:\n"
            f"  {error}\n"
            f"The function must be syntactically correct Python and only use "
            f"the provided API (predicates, actions, pd, problem, domain, planner)."
        )

    elif ft == "search_error":
        error = feedback.get("error", "(unknown error)")
        return (
            f"The planning search encountered an unexpected error:\n"
            f"  {error}\n"
            f"This may indicate a problem with the constraint function or "
            f"the planning infrastructure. Please revise the constraints."
        )

    else:
        return "The previous constraints did not lead to a successful plan."

def _format_feedback_list(feedback_list: list) -> str:
    """Format a list of feedback dicts (one per iteration) into a numbered summary."""
    if not feedback_list:
        print("[vrm.py] NO FEEDBACK FOUND")
        return (
            "No structured feedback entries were recorded for these iterations. "
            "The previous constraints still failed to produce a valid plan."
        )
    lines = []
    for i, fb in enumerate(feedback_list):
        lines.append(f"  Iteration {i + 1}: {_format_feedback(fb)}")
    return "\n".join(lines)

def feedback_prompt(feedback_list: list, problem_description: str = None) -> str:
    num_failed_iterations = len(feedback_list)
    feedback_summary = _format_feedback_list(feedback_list)
    return textwrap.dedent(f"""\t\tYour constraints failed to produce a valid plan in the previous {num_failed_iterations} iterations.
    The description for the current problem instance is:
    {problem_description}
    Here is what happened in each iteration:
    {feedback_summary}
    Revise your constraints function accordingly.
    Add comments to explain your constraints.
    {REMINDERS}
    """)

def append_file(source_path, destination_path):
    with open(source_path, 'r') as source_file:
        content = source_file.read()

    with open(destination_path, 'a') as dest_file:
        dest_file.write('\n')  # Optional: add newline between files
        dest_file.write(content)

def post_process_code(response):
    # stop stupid gemini from putting text at the top of the code
    final_code = response.strip()
    final_code = final_code[final_code.find("def generate_con"):]

    # find and remove any code fences
    final_code = re.sub(r"```python\n", "", final_code)
    final_code = re.sub(r"```", "", final_code)

    # add final function call only if not already present
    if not final_code.rstrip().endswith("generate_constraints(problem, domain, planner, pd)"):
        final_code = final_code + "\n\ngenerate_constraints(problem, domain, planner, pd)"

    # Last-pass syntax scrub: if Python reports a bad line, drop it and retry.
    # This is defensive cleanup for occasionally malformed model output.
    lines = final_code.splitlines()
    while lines:
        candidate = "\n".join(lines)
        try:
            compile(candidate, "<generated_constraints>", "exec")
            final_code = candidate
            break
        except SyntaxError as e:
            bad_line = getattr(e, "lineno", None)
            if bad_line is None or bad_line < 1 or bad_line > len(lines):
                print(f"[post_process_code] Syntax check failed but no removable line: {e}")
                final_code = candidate
                break
            removed = lines.pop(bad_line - 1)
            print(
                f"[post_process_code] Removed syntax-error line {bad_line}: "
                f"{removed.strip()}"
            )
    return final_code

def run_vrm(
    exp: str,
    provider: str,
    iteration: int = 0,
    feedback: bool = False,
    model: str = None,
    reasoning_effort: str = None,
    reprompt_failure_threshold: int = 1,
    prompt_number: int = 0, # how many prompts have been run so far
):

    client_kwargs = {}
    if model:
        client_kwargs["model"] = model
    if reasoning_effort is not None:
        client_kwargs["reasoning_effort"] = reasoning_effort
    client = get_vlm_client(provider, **client_kwargs)
    start = time.time()
    dead_time = 0.0  # time lost to failed server-overload attempts + retry waits

    if exp == "shelf":
        grasp_info = "grasp the SIDE of the object, approaching grid from negative Y direction, end-effector at fixed height throughout motion"
    elif exp == "rack":
        grasp_info = "grasp the TOP of the object, hover directly above the target position and descend straight down along z; the object never moves sideways while it is near the rack"
    elif exp in ("containers", "packing", "cooking"):
        grasp_info = "grasp the TOP of the object"
    else:
        grasp_info = "grasp the TOP of the object (default)"

    base_path = f"envs/{exp}"
    input_path = os.path.join(base_path, "prompting", "input")
    output_path = os.path.join(base_path, "prompting", "output")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    scene_description = str(_read_prompt(os.path.join(input_path, "scene_description.txt")))
    domain_pddl_path = resolve_generated_pddl(os.path.join(base_path, "domain_z3.pddl"))
    domain_description = str(_read_prompt(domain_pddl_path))
    problem_description = str(_read_prompt(os.path.join(base_path, "temp_problem.pddl")))
    starter_code = str(_read_prompt(os.path.join(input_path, "starter.txt")))

    with open(os.path.join(input_path, "default_camera_capture.png"), "rb") as image_file:
        scene_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Continue prior conversation only when feedback prompting is enabled.
    use_feedback = bool(feedback) and (prompt_number > 0)

    if not use_feedback:
        print(f"[run_vrm] NEW CONVERSATION (prompt_number: {prompt_number})")

        prompt1 = textwrap.dedent(f"""\t\tI am performing task-and-motion planning in this environment.
        My robot arm grasps objects according to this specification: {grasp_info}
        The textual scene description:\n{scene_description}
        The PDDL domain:\n{domain_description}
        The PDDL problem:\n{problem_description}
        Could a task-level plan generated under this domain fail during downward refinement?
        If so, describe the general failure mode(s) precisely.
        If there is only one failure mode, describe it in a single sentence.
        If there are multiple failure modes, list each in a single sentence.
        If no failure is expected, say 'No failure expected.'
        """)

        prompt2 = textwrap.dedent(f"""\t\tWe must prevent these failure modes by adding constraints to our task-level planner.
        Constraints should be general, and apply to any problem instance under this domain.
        Define a function which will be called before planning to apply constraints within a SAT-based planner.
        Follow this reference code and instructions:
        {starter_code}
        {REMINDERS}
        """)
        with open(os.path.join(output_path, "initial_prompt_1.txt"), "w", encoding="utf-8") as f:
            f.write(prompt1)
        with open(os.path.join(output_path, "initial_prompt_2.txt"), "w", encoding="utf-8") as f:
            f.write(prompt2)

        print("[run_vrm] Running Prompt 1")
        while True:
            call_start = time.time()
            try:
                response1 = client.generate(text=prompt1, images=[scene_image])
                break
            except Exception as e:
                if _is_server_overload(e):
                    dead_time += time.time() - call_start
                    print(f"[run_vrm] Server overload on Prompt 1: {e}")
                    print(f"[run_vrm] Waiting {_SERVER_RETRY_WAIT}s before retrying...")
                    time.sleep(_SERVER_RETRY_WAIT)
                    dead_time += _SERVER_RETRY_WAIT
                    continue
                raise
        with open(os.path.join(output_path, "response_1.txt"), "w", encoding="utf-8") as f:
            f.write(response1)

        print("[run_vrm] ===== Response 1: potential failure modes =====")
        print(response1.strip())
        print("[run_vrm] ==============================================")

        print("[run_vrm] Running Prompt 2")
        while True:
            call_start = time.time()
            try:
                response2 = client.generate_multiturn(
                    messages=[
                        {"role": "user", "text": prompt1},
                        {"role": "assistant", "text": response1},
                        {"role": "user", "text": prompt2},
                    ],
                    images=[scene_image],
                )
                break
            except Exception as e:
                if _is_server_overload(e):
                    dead_time += time.time() - call_start
                    print(f"[run_vrm] Server overload on Prompt 2: {e}")
                    print(f"[run_vrm] Waiting {_SERVER_RETRY_WAIT}s before retrying...")
                    time.sleep(_SERVER_RETRY_WAIT)
                    dead_time += _SERVER_RETRY_WAIT
                    continue
                raise

        final_code = post_process_code(response2)
        # In no-feedback mode we still version constraints by prompt_number
        # so the planner can load constraints_{prompt_number}.py correctly.
        with open(os.path.join(output_path, f"constraints_{prompt_number}.py"), "w", encoding="utf-8") as f:
            f.write(final_code)

    else:
        print(f"[run_vrm] CONTINUING CONVERSATION (prompt_number: {prompt_number})")
        # Continue from previous conversations during the same run.
        # The conversation persists across one run of run.py,
        # over potentially several problems each with multiple iterations.

        # Read the initial prompts and responses
        with open(os.path.join(output_path, "initial_prompt_1.txt"), "r", encoding="utf-8") as f:
            init_prompt1 = f.read()
        with open(os.path.join(output_path, "initial_prompt_2.txt"), "r", encoding="utf-8") as f:
            init_prompt2 = f.read()
        with open(os.path.join(output_path, "response_1.txt"), "r", encoding="utf-8") as f:
            init_response1 = f.read()

        # Build the conversation context starting with the initial exchange
        context = [
            {"role": "user", "text": init_prompt1},
            {"role": "assistant", "text": init_response1},
            {"role": "user", "text": init_prompt2},
        ]

        # Append each previous feedback/constraints exchange.
        # There should be prompt_number constraints files (constraints_0..constraints_{prompt_number-1})
        # and prompt_number feedback files (feedback_0..feedback_{prompt_number-1}).
        # Each feedback file contains a list of feedback dicts, one per planning iteration.
        for i in range(prompt_number):
            feedback_path = os.path.join(output_path, f"feedback_{i}.json")
            constraints_path = os.path.join(output_path, f"constraints_{i}.py")

            with open(feedback_path, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)

            with open(constraints_path, "r", encoding="utf-8") as f:
                constraints_text = f.read()

            # For i == 0 the assistant turn is already in context (init_constraints),
            # so we only need the user feedback turn for i == 0, then both for i > 0.
            context.append({"role": "assistant", "text": constraints_text})
            context.append({"role": "user", "text": feedback_prompt(feedback_list, problem_description)})


        # dump all context into a file
        with open(os.path.join(output_path, "dump.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(context, indent=2, ensure_ascii=False))

        # If prompt_number > 1, the last assistant turn (constraints_{prompt_number-1})
        # is already appended as the response to the previous feedback.
        # We need the final feedback turn for the current prompt_number.
        # Actually, re-think: the loop above builds history for prompts 0..prompt_number-1.
        # For i=0: we add user feedback_0 (assistant constraints_0 is already in init context).
        # For i>0: we add user feedback_i + assistant constraints_i.
        # The model will now generate constraints_{prompt_number} as the response.

        print(f"[run_vrm] Running feedback prompt {prompt_number} (conversation has {len(context)} turns)")
        while True:
            call_start = time.time()
            try:
                response = client.generate_multiturn(
                    messages=context,
                    images=[scene_image],
                )
                break
            except Exception as e:
                if _is_server_overload(e):
                    dead_time += time.time() - call_start
                    print(f"[run_vrm] Server overload on feedback prompt: {e}")
                    print(f"[run_vrm] Waiting {_SERVER_RETRY_WAIT}s before retrying...")
                    time.sleep(_SERVER_RETRY_WAIT)
                    dead_time += _SERVER_RETRY_WAIT
                    continue
                raise

        final_code = post_process_code(response)
        with open(os.path.join(output_path, f"constraints_{prompt_number}.py"), "w", encoding="utf-8") as f:
            f.write(final_code)


    end = time.time()
    elapsed = end - start - dead_time
    if dead_time > 0:
        print(f"[run_vrm] Elapsed time: {elapsed:.4f} seconds (excluded {dead_time:.1f}s of server-overload retry overhead)")
    else:
        print(f"[run_vrm] Elapsed time: {elapsed:.4f} seconds")
