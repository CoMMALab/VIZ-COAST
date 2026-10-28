#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import os
import pathlib
import random
import time
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple, Any
import sys

import numpy as np
import pybullet_tools as pbt
from pybullet_tools import kuka_primitives, utils as pb_utils
import spatialdyn as dyn
import symbolic

import coast
from coast.constraint import Constraint


@dataclasses.dataclass
class World:
    robot: int
    ab: dyn.ArticulatedBody
    body_names: Dict[int, str]
    body_types: Dict[int, str]
    movable_bodies: List[int]
    fixed_bodies: List[Any]
    new_state: Set[str]
    q_home: np.ndarray


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_trial_results_file(
    cfg_path: str, results_file: Optional[str], trial_mode: bool
) -> Optional[str]:
    if not trial_mode or not results_file:
        return results_file
    cfg_file_name = pathlib.Path(str(cfg_path)).stem
    trial_dir = pathlib.Path(results_file) / cfg_file_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    while (trial_dir / f"{n}.csv").exists():
        n += 1
    return str(trial_dir / f"{n}.csv")


def _parse_xyz_attr(value: Optional[str]) -> Tuple[float, float, float]:
    if not value:
        return 0.0, 0.0, 0.0
    xyz = [float(v) for v in value.split()]
    if len(xyz) != 3:
        return 0.0, 0.0, 0.0
    return xyz[0], xyz[1], xyz[2]


def _parse_urdf_container_dims(urdf_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (external_hint, internal_hint) dimensions from URDF collision boxes."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    base_size = None
    base_origin = (0.0, 0.0, 0.0)
    wall_x_thickness = None
    wall_y_thickness = None
    wall_top_z = None

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        collision = link.find("collision")
        if collision is None:
            continue
        origin_node = collision.find("origin")
        origin = _parse_xyz_attr(
            origin_node.attrib.get("xyz") if origin_node is not None else None
        )
        box = collision.find("geometry/box")
        if box is None:
            continue
        size = _parse_xyz_attr(box.attrib.get("size"))
        size_arr = np.array(size, dtype=float)

        if link_name == "base_link":
            base_size = size_arr
            base_origin = origin
            continue

        if "front" in link_name or "back" in link_name:
            wall_y_thickness = size_arr[1]
        if "left" in link_name or "right" in link_name:
            wall_x_thickness = size_arr[0]

        this_top = origin[2] + size_arr[2] / 2.0
        wall_top_z = this_top if wall_top_z is None else min(wall_top_z, this_top)

    if base_size is None:
        return None, None

    # Fallback thickness estimates if a wall direction is missing from URDF.
    if wall_x_thickness is None:
        wall_x_thickness = 0.01
    if wall_y_thickness is None:
        wall_y_thickness = 0.01

    internal_x = max(0.0, base_size[0] - 2.0 * wall_x_thickness)
    internal_y = max(0.0, base_size[1] - 2.0 * wall_y_thickness)
    base_top_z = base_origin[2] + base_size[2] / 2.0
    if wall_top_z is None:
        internal_z = 0.0
    else:
        internal_z = max(0.0, wall_top_z - base_top_z)

    external_hint = base_size
    internal_hint = np.array([internal_x, internal_y, internal_z], dtype=float)
    return external_hint, internal_hint


def _print_dimension_report(
    block_type_dims: Dict[str, Tuple[float, float, float, Any]],
    container_urdf_path: str,
    container_body: int,
) -> None:
    print("[packing.load_world] Object dimensions (x, y, z):")
    for pddl_type in ("slab", "bblock", "sblock"):
        x, y, z, _ = block_type_dims[pddl_type]
        print(f"  - {pddl_type}: ({x:.3f}, {y:.3f}, {z:.3f})")

    aabb = pb_utils.get_aabb(container_body)
    external = np.array(aabb.upper) - np.array(aabb.lower)
    urdf_external, urdf_internal = _parse_urdf_container_dims(container_urdf_path)
    print(
        f"  - container external (AABB): ({external[0]:.3f}, {external[1]:.3f}, {external[2]:.3f})"
    )
    if urdf_external is not None and urdf_internal is not None:
        print(
            f"  - container external (URDF base): ({urdf_external[0]:.3f}, {urdf_external[1]:.3f}, {urdf_external[2]:.3f})"
        )
        print(
            f"  - container internal (URDF cavity): ({urdf_internal[0]:.3f}, {urdf_internal[1]:.3f}, {urdf_internal[2]:.3f})"
        )


def _block_names_and_types(num_blocks: list) -> List[Tuple[str, str]]:
    """Generate (name, pddl_type) pairs for blocks given [num_slab, num_bblock, num_sblock]."""
    result = []
    for i in range(num_blocks[0]):
        result.append((f"slab{i + 1}", "slab"))
    for i in range(num_blocks[1]):
        result.append((f"bb{i + 1}", "bblock"))
    for i in range(num_blocks[2]):
        result.append((f"sb{i + 1}", "sblock"))
    return result


def _scene_color_for_type(pddl_type: str) -> str:
    if pddl_type == "slab":
        return "yellow"
    if pddl_type == "bblock":
        return "green"
    if pddl_type == "sblock":
        return "blue"
    if pddl_type == "lid":
        return "gray"
    if pddl_type == "box":
        return "red"
    return "gray"


def write_scene_description(world: World) -> None:
    scene_path = pathlib.Path("envs/packing/prompting/input/scene_description.txt")
    scene_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["Here is the scene description. x and y correspond to plane coordinates:", "Objects:"]
    container_urdf = pathlib.Path("models/tetris_box.urdf")
    container_internal_dims = None
    if container_urdf.exists():
        _, container_internal_dims = _parse_urdf_container_dims(str(container_urdf))

    for body_id, name in world.body_names.items():
        pddl_type = world.body_types[body_id]
        if pddl_type == "static":
            continue
        point, _ = pb_utils.get_pose(body_id)
        aabb = pb_utils.get_aabb(body_id)
        ext = np.array(aabb.upper) - np.array(aabb.lower)
        ext_dims = f"external_dims(x={ext[0]:.3f}, y={ext[1]:.3f}, z={ext[2]:.3f})"
        dims_field = ext_dims
        if pddl_type == "box" and container_internal_dims is not None:
            dims_field = (
                f"{ext_dims}, "
                f"internal_dims(x={container_internal_dims[0]:.3f}, "
                f"y={container_internal_dims[1]:.3f}, z={container_internal_dims[2]:.3f})"
            )
        lines.append(
            f'{name}{{type:"{pddl_type}",color:"{_scene_color_for_type(pddl_type)}",'
            f"location(x={point[0]:.3f}, y={point[1]:.3f}),{dims_field}}}"
        )
    lines.append("")
    lines.append("robot{location(x=0,y=0)}")
    scene_path.write_text("\n".join(lines) + "\n")


# num_blocks takes the form [num_slab, num_bblock, num_sblock]
def load_world(num_blocks: list) -> World:
    pbt.utils.set_default_camera()
    # pbt.utils.draw_global_system()

    cwd = os.getcwd()
    print(f"cwd: {cwd}")
    with pbt.utils.HideOutput():
        external_models_dir = "models"
        internal_models_dir = f"{cwd}/models"

        table = pbt.utils.load_model(f"{internal_models_dir}/long_floor.urdf")
        robot = pbt.utils.load_model(
            pbt.utils.DRAKE_IIWA_URDF,
            pose=pb_utils.Pose(pb_utils.Point(y=+0.0)),
        )

        ab = dyn.urdf.load_model(
            str(
                pathlib.Path("external/pybullet-planning/")
                / pbt.utils.DRAKE_IIWA_URDF
            )
        )

        box_coords = [0.0, 0.3]
        box1 = pbt.utils.load_model(
            f"{internal_models_dir}/tetris_box.urdf",
            pose=pb_utils.Pose(pb_utils.Point(box_coords[0], box_coords[1])),
        )

        lid1 = pbt.utils.load_model(
            f"{internal_models_dir}/lid.urdf",
            pose=pb_utils.Pose(pb_utils.Point(x=0.5, y=0.3, z=0.01)),
        )

        body_names: Dict[int, str] = {
            table: "table",
            box1: "box1",
            lid1: "lid1",
        }
        body_types: Dict[int, str] = {
            table: "static",
            box1: "box",
            lid1: "lid",
        }

        # Block dimensions: (width, width, height, color)
        # sblock widened to ~match end-effector wrist flange (~0.07);
        # bblock thinned slightly (0.10 → 0.09) for clearance.
        type_dims = {
            "slab": (0.2, 0.2, 0.05, pbt.utils.YELLOW),
            "bblock": (0.09, 0.09, 0.1, pbt.utils.GREEN),
            "sblock": (0.06, 0.06, 0.05, pbt.utils.BLUE),
        }
        container_urdf = str(
            pathlib.Path(internal_models_dir) / "tetris_box.urdf"
        )
        _print_dimension_report(type_dims, container_urdf, box1)

        block_name_types = _block_names_and_types(num_blocks)
        blocks = []
        for name, pddl_type in block_name_types:
            w1, w2, h, color = type_dims[pddl_type]
            block = pbt.utils.create_box(w1, w2, h, color=color)
            blocks.append(block)
            body_names[block] = name
            body_types[block] = pddl_type

    pb_utils.draw_pose(
        pb_utils.Pose(),
        parent=robot,
        parent_link=kuka_primitives.get_tool_link(robot),
    )

    fixed_bodies = [table, box1]
    movable_bodies = blocks + [lid1]

    # Set lid initial position
    pb_utils.set_pose(
        lid1,
        pb_utils.Pose(pb_utils.Point(0.5, 0.3, z=pb_utils.stable_z(lid1, table))),
    )

    # Initial positions grouped by type: [slabs, bblocks, sblocks]
    # Keep packables in front of the robot (positive y), but off to the side
    # so they don't start behind the arm or opposite the container.
    block_init_positions = [
        [  # slabs
            (-0.60, 0.0),
            (-0.35, 0.0),
            (-0.35, -0.25),
        ],
        [  # bblocks
            (-0.60, 0.2),
            (-0.42, 0.2),
            (-0.24, 0.2),
        ],
        [  # sblocks
            (-0.60, 0.35),
            (-0.50, 0.35),
            (-0.40, 0.35),
        ],
    ]

    idx = 0
    for type_idx in range(3):
        for j in range(num_blocks[type_idx]):
            pb_utils.set_pose(
                blocks[idx],
                pb_utils.Pose(
                    pb_utils.Point(
                        block_init_positions[type_idx][j][0],
                        block_init_positions[type_idx][j][1],
                        z=pb_utils.stable_z(blocks[idx], table),
                    )
                ),
            )
            idx += 1

    # Build PDDL initial state
    new_state: Set[str] = set()
    for m in movable_bodies:
        new_state.add(f"(on {body_names[m]} table)")
    new_state.add("(packed box1 box1)")  # base case for pack action

    q_home = np.array([0.0, -np.pi / 4, 0.0, -np.pi / 2, 0.0, np.pi / 3, 0.0])
    world = World(
        robot,
        ab,
        body_names,
        body_types,
        movable_bodies,
        fixed_bodies,
        new_state,
        q_home,
    )
    # Keep prompting inputs in sync with the generated scene.
    write_scene_description(world)
    return world


def init_geometric_state_tetris(
    num_blocks: list, use_gui: bool = True
) -> Tuple[World, Any, Set[str], List[Any]]:
    pbt.utils.connect(use_gui=use_gui)
    world = load_world(num_blocks)

    state = coast.GeometricState()
    objects: List[coast.Object] = [
        coast.Object(
            name=body_name,
            object_type=world.body_types[body_id],
            value=body_id,
        )
        for body_id, body_name in world.body_names.items()
    ]

    stream_state: Set[coast.CertifiedFact] = set()
    state.set(
        "config",
        0,
        (
            world.robot,
            pbt.kuka_primitives.BodyConf(
                world.robot, pbt.utils.get_configuration(world.robot)
            ),
        ),
    )
    state.set("cur_grasp", 0, None)
    state.set("commands", 0, [])
    stream_state.add("AtConf(q0)")
    objects.append(
        coast.Object(
            "q0",
            "conf",
            pbt.kuka_primitives.BodyConf(
                world.robot, pbt.utils.get_configuration(world.robot)
            ),
        )
    )

    for body_id in world.movable_bodies:
        body_pose = pbt.kuka_primitives.BodyPose(body_id, pbt.utils.get_pose(body_id))
        state.set(world.body_names[body_id], 0, (body_id, body_pose))
        stream_state.add(
            f"AtPose({world.body_names[body_id]}, p0_{body_id})"
        )
        objects.append(coast.Object(f"p0_{body_id}", "pose", body_pose))

    return world, state, stream_state, objects


def create_problem(
    world: World,
    dynamic_timesteps: bool = False,
    timestep_count: Optional[int] = None,
    include_timesteps: bool = True,
) -> str:
    """Build the PDDL problem string and generate temp_domain.pddl with timesteps.

    Returns the problem PDDL string (also written to envs/packing/temp_problem.pddl).
    """
    # --- Compute timestep count ---
    if timestep_count is None:
        if dynamic_timesteps:
            # Number of packable items (slab, bblock, sblock — everything
            # except static objects, boxes, and lids).
            num_packables = sum(
                1 for bt in world.body_types.values()
                if bt in ("slab", "bblock", "sblock")
            )
            timestep_count = num_packables * 2 + 3
        else:
            timestep_count = 30
    timestep_count = max(2, int(timestep_count))

    # --- Build objects section (exclude domain constants like table) ---
    domain_constants = {
        "box1",
        "lid1",
        "slab1",
        "slab2",
        "slab3",
        "bb1",
        "bb2",
        "bb3",
        "sb1",
        "sb2",
        "sb3",
    }
    objects_lines = []
    for body_id, name in world.body_names.items():
        pddl_type = world.body_types[body_id]
        if name == "table" or name in domain_constants:
            continue  # table is a domain constant
        objects_lines.append(f"        {name} - {pddl_type}")
    objects_str = "\n".join(objects_lines)

    # --- Build init section from world.new_state (+ timesteps if needed) ---
    init_lines = []
    for prop in sorted(world.new_state):
        init_lines.append(f"        {prop}")
    if include_timesteps:
        init_lines.append("        (AtTimestep t1)")
        for t in range(timestep_count - 1):
            init_lines.append(f"        (Next t{t + 1} t{t + 2})")
    init_str = "\n".join(init_lines)

    # --- Grounded goal: all packable items packed, all lids on a box ---
    # We enumerate explicitly because symbolic.Problem can't serialize quantified goals.
    goal_parts = []
    for body_id, name in world.body_names.items():
        ptype = world.body_types[body_id]
        if ptype in ("slab", "bblock", "sblock"):
            goal_parts.append(f"(packed {name} box1)")
        elif ptype == "lid":
            goal_parts.append(f"(on {name} box1)")
    goal = "(and\n" + "\n".join(f"        {g}" for g in goal_parts) + "\n    )"

    problem_pddl = f"""(define (problem tetris-pack)
    (:domain pack)
    (:objects
{objects_str}
    )
    (:init
{init_str}
    )
    (:goal {goal})
)
"""

    with open("envs/packing/temp_problem.pddl", "w") as f:
        f.write(problem_pddl)

    # --- Generate temp_domain.pddl with timestep constants injected (COAST mode) ---
    if include_timesteps:
        with open("envs/packing/domain_coast.pddl", "r") as f:
            domain_str = f.read()

        timesteps_insertion = ""
        for i in range(timestep_count):
            timesteps_insertion += f"\n        t{i + 1} - timestep"
        domain_str = domain_str.replace(";timesteps here", timesteps_insertion)

        with open("envs/packing/temp_domain.pddl", "w") as f:
            f.write(domain_str)

    return problem_pddl


def main(
    cfg: Any,
    results_file: str,
    streams_py: str,
    algorithm: str,
    max_level: int,
    pybullet: bool,
    search_sample_ratio: float,
    timeout: int,
    reps: int,
    random_seed: int,
    append: bool,
    use_gui: bool,
    skip_execution: bool,
    no_write: bool,
    continue_from: Optional[str] = None,
    show_thinking: bool = False,
):

    # Remove stale output files so the VLM is prompted fresh on the first experiment.
    # Oracles live in ../oracles/, outside this directory, so they are untouched.
    _output_dir = "envs/packing/prompting/output"
    if os.path.isdir(_output_dir):
        for _fname in os.listdir(_output_dir):
            _fpath = os.path.join(_output_dir, _fname)
            if os.path.isfile(_fpath):
                os.remove(_fpath)

    # json should contain a list of dicts, where each dict is a different experiment configuration
    experiments = json.load(open(cfg))

    if type(experiments) == dict:
        # we have global settings
        global_settings = experiments['global']
        experiments = experiments['specific']
        if reps == 0:
            reps = global_settings['reps']
        if results_file is None:
            results_file = global_settings['results_file']
    else:
        global_settings = None

    trial_mode = _as_bool(
        global_settings.get("trial_mode", False) if global_settings else False
    )
    if not global_settings and len(experiments) > 0:
        trial_mode = _as_bool(experiments[0].get("trial_mode", False))

    resolved_results_file = _resolve_trial_results_file(cfg, results_file, trial_mode)
    if trial_mode and resolved_results_file:
        append = False
        print(f"[packing.run] trial_mode output: {resolved_results_file}")

    # Initialize CSV file - headers will be written with first result
    csv_initialized = False
    csv_fieldnames = None  # locked on first write to keep column order stable

    # --continue-from: resume appending to an existing CSV, skipping
    # experiments that have already been written (by row count).
    rows_to_skip = 0
    if continue_from is not None:
        if not os.path.exists(continue_from):
            raise FileNotFoundError(f"--continue-from file not found: {continue_from}")
        resolved_results_file = continue_from
        append = True
        csv_initialized = True  # header already present
        with open(continue_from, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # read header
            if header:
                csv_fieldnames = header  # lock to existing column order
            rows_to_skip = sum(1 for _ in reader)
        print(
            f"[packing.run] Continuing from {continue_from}, "
            f"skipping {rows_to_skip} completed experiment(s)"
        )

    experiment_counter = 0
    for cfg_idx, config in enumerate(experiments):
        if global_settings:
            for key, value in global_settings.items():
                if key not in config:
                    config[key] = value
                else:
                    print(f"Warning: {key} already in config, skipping")

        num_blocks = config['num_blocks']

        if config.get("skip", False):
            continue

        for e in range(reps):
            # --continue-from: skip experiments already recorded in the CSV.
            if experiment_counter < rows_to_skip:
                experiment_counter += 1
                print(f"[packing.run] Skipping experiment {experiment_counter}/{rows_to_skip} "
                      f"(cfg_idx={cfg_idx}, rep={e}, num_blocks={num_blocks})")
                continue
            experiment_counter += 1

            if random_seed is not None:
                np.random.seed(int(random_seed))
                random.seed(int(random_seed))
                print("random_seed args:", random_seed)
            else:
                print("random_seed:", e)
                np.random.seed(e)
                random.seed(e)

            (
                world,
                state,
                stream_state,
                objects,
            ) = init_geometric_state_tetris(num_blocks, use_gui=use_gui)

            # Build PDDL files from world state (must happen after world init)
            method = config.get("method", "coast_z3")
            dynamic_timesteps = bool(config.get("dynamic_timesteps", False))
            timestep_count = config.get("timestep_count")
            use_timesteps = method.lower() in ("coast_fd", "coast_z3")
            problem_pddl = create_problem(
                world,
                dynamic_timesteps=dynamic_timesteps,
                timestep_count=timestep_count,
                include_timesteps=use_timesteps,
            )
            if method.lower() in ("coast_fd", "coast_z3"):
                # Both coast_fd and coast_z3 share the same domain
                # (same :adl requirements and type hierarchy).
                domain_pddl = "envs/packing/temp_domain.pddl"  # built from domain_coast.pddl
            else:
                domain_pddl = "envs/packing/domain_z3.pddl"

            # Print the VLM's chain-of-thought to the terminal (VLMB method only).
            config['show_thinking'] = config.get('show_thinking', show_thinking)

            saver = pbt.utils.WorldSaver()
            pbt.utils.set_renderer(enable=True)
            camera_pose = [-1.2, 1.2, 1]
            target_pos = [0, 0, 0.4]
            image = pbt.utils.get_image(
                camera_pose, target_pos, width=1024, height=768, vertical_fov=45.0
            )
            pbt.utils.save_image(
                "envs/packing/prompting/input/default_camera_capture.png", image.rgbPixels
            )

            plan = coast.plan(
                cfg=config,
                domain_pddl=domain_pddl,
                problem_pddl=problem_pddl,
                streams_py=streams_py,
                algorithm=algorithm,
                max_level=max_level,
                search_sample_ratio=search_sample_ratio,
                timeout=timeout,
                experiment=None,
                stream_state=stream_state,
                objects=objects,
                random_seed=random_seed,
                world=world,
                constraint_cls=Constraint,
                sort_streams=False,
                use_cache=False,
            )
            print("PLANNING DONE")

            # Prepare result data
            res = plan.log.get_timing()
            res['success'] = True
            res['rep_idx'] = e
            res['num_blocks'] = num_blocks
            if plan.action_plan is None or plan.objects is None:
                print("No plan found")
                res['success'] = False

            # Write to CSV line-by-line
            if not no_write:
                file_mode = "a" if csv_initialized or append else "w"
                # Lock column order on first write so that later rows
                # (whose dict key order may differ) stay aligned with
                # the header.
                if csv_fieldnames is None:
                    csv_fieldnames = list(res.keys())
                with open(resolved_results_file, file_mode, newline="") as f_object:
                    writer_csv = csv.DictWriter(
                        f_object, fieldnames=csv_fieldnames
                    )

                    # Write header only with the first result
                    if not csv_initialized:
                        writer_csv.writeheader()
                        csv_initialized = True

                    writer_csv.writerow(res)

            if res['success'] and not skip_execution:
                pbt.utils.set_renderer(enable=True)
                saver.restore()
                plan.log.print()
                if pybullet:
                    pbt.utils.wait_for_user("Execute?")
                    time.sleep(1)
                    for action in plan.action_plan:
                        action.execute(plan.objects)
                        if str(action)[0].lower() == 'c':
                            time.sleep(0.3)
            time.sleep(1)
            if pbt.utils.is_connected():
                pbt.utils.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--streams-py",
        nargs="?",
        default="envs/packing/streams.py",
        help="Python streams file.",
    )
    parser.add_argument(
        "--algorithm",
        nargs="?",
        default="improved",
        help="Planning algorithm to use",
    )
    parser.add_argument(
        "--max-level",
        nargs="?",
        type=int,
        default=6,
        help="Max planning level.",
    )
    parser.add_argument(
        "--pybullet",
        type=bool,
        default=True,
        help="Running a domain that runs pybullet",
    )
    parser.add_argument(
        "--search-sample-ratio",
        type=float,
        default=10,
        help="The ratio of search versus sample",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout for algorithm"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=0,
        help="The number of times you want to run each experiment configuration",
    )
    parser.add_argument(
        "--results-file",
        type=Any,
        default=None,
        help="The file to write the csv of experiment",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="envs/packing/config.json",
        help="Configuration file",
    )
    parser.add_argument("--random-seed", type=int, default=3, help="Random seed")
    parser.add_argument("--no-write", action="store_true", help="Skip writing to the results file")
    parser.add_argument("--append", action="store_true", help="Append to the results file")
    parser.add_argument("--use-gui", action="store_true", help="Use GUI")
    parser.add_argument("--skip-execution", action="store_true", help="Skip execution")
    parser.add_argument("--continue-from", type=str, default=None, help="A csv file to continue from")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Print the VLM's chain-of-thought to the terminal (VLMB method only).")
    args = parser.parse_args()
    try:
        main(**vars(args))
    except Exception as e:
        if pbt.utils.is_connected():
            pbt.utils.disconnect()
        raise e
