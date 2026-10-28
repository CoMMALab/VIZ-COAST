#!/usr/bin/env python3

import faulthandler
faulthandler.enable()  # Print traceback on segfault instead of silent crash

import argparse
import csv
import dataclasses
import os
import pathlib
import re
import random
import time
from typing import Dict, List, Optional, Set, Tuple
import json
import numpy as np
import pybullet_tools as pbt
import pybullet_tools.kuka_primitives
import spatialdyn as dyn
import symbolic


import coast
from constraints import LocationConstraint
from coast.constraint import Constraint


@dataclasses.dataclass
class World:
    robot: int
    ab: dyn.ArticulatedBody
    q_home: np.ndarray
    body_names: Dict[int, str]
    movable_bodies: List[int]
    fixed_bodies: List[int]
    table: int
    block_height: float
    new_state: Set[str]
    locations: List[Optional[pbt.utils.Pose]]
    q0: int
    grid_rows: int
    grid_cols: int
    goal_loc: int  # 0-indexed location index of the red block's goal
    shelf_bodies: List[int]  # transparent walls + lid enclosing the grid (excludes table)
    shelf_retract_y: Optional[float]  # y-coord IK approach poses clamp to for full retraction; None if no shelf


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


def write_scene_description(world: World, num_blocks: int) -> None:
    scene_path = pathlib.Path("envs/shelf/prompting/input/scene_description.txt")
    scene_path.parent.mkdir(parents=True, exist_ok=True)

    block_locations: Dict[str, str] = {}
    for prop in world.new_state:
        match = re.match(r"\(ontable\s+(b\d+)\s+(loc\d+)\)", prop)
        if match:
            block_locations[match.group(1)] = match.group(2)

    def _xy_from_pose_like(pose_like):
        if hasattr(pose_like, "point"):
            point = pose_like.point
            return float(point[0]), float(point[1])
        if isinstance(pose_like, (tuple, list)) and len(pose_like) >= 1:
            point = pose_like[0]
            if isinstance(point, (tuple, list, np.ndarray)) and len(point) >= 2:
                return float(point[0]), float(point[1])
        raise ValueError(f"Unsupported pose format for scene description: {type(pose_like)}")

    lines = ["Here is the scene description. x and y correspond to plane coordinates:", "Objects:"]
    for idx, loc in enumerate(world.locations):
        if loc is None:
            continue
        x, y = _xy_from_pose_like(loc)
        lines.append(
            f'tile{idx + 1}{{color:"green",location(x={x:.3f}, y={y:.3f}),id:{idx + 1}}}'
        )

    lines.append("")
    for idx, loc in enumerate(world.locations):
        if loc is None:
            continue
        x, y = _xy_from_pose_like(loc)
        lines.append(f"loc{idx + 1}{{x={x:.3f}, y={y:.3f}}}")

    table_point, _ = pbt.utils.get_pose(world.table)
    table_aabb = pbt.utils.get_aabb(world.table)
    table_ext = np.array(table_aabb.upper) - np.array(table_aabb.lower)
    lines.append("")
    lines.append(
        f'mytable{{type:"surface",color:"gray",location(x={table_point[0]:.3f}, y={table_point[1]:.3f}),'
        f'external_dims(x={table_ext[0]:.3f}, y={table_ext[1]:.3f}, z={table_ext[2]:.3f})}}'
    )

    def _block_idx(body_id: int) -> int:
        return int(world.body_names[body_id][1:])

    for body_id in sorted(world.movable_bodies, key=_block_idx):
        block_name = world.body_names[body_id]
        point, _ = pbt.utils.get_pose(body_id)
        aabb = pbt.utils.get_aabb(body_id)
        ext = np.array(aabb.upper) - np.array(aabb.lower)
        color = "red" if block_name == f"b{num_blocks}" else "blue"
        grid_loc = block_locations.get(block_name, "unknown")
        lines.append(
            f'{block_name}{{type:"block",color:"{color}",location(x={point[0]:.3f}, y={point[1]:.3f}),'
            f'id:"{block_name}",grid_location:"{grid_loc}",'
            f'external_dims(x={ext[0]:.3f}, y={ext[1]:.3f}, z={ext[2]:.3f})}}'
        )

    lines.append("")
    lines.append("robot{location(x=0,y=0)}")
    scene_path.write_text("\n".join(lines) + "\n")


def write_template_scene_description(world: World) -> None:
    template_path = pathlib.Path("envs/shelf/prompting/input/template_scene_description.txt")
    template_path.parent.mkdir(parents=True, exist_ok=True)

    def _xy_from_pose_like(pose_like):
        if hasattr(pose_like, "point"):
            point = pose_like.point
            return float(point[0]), float(point[1])
        if isinstance(pose_like, (tuple, list)) and len(pose_like) >= 1:
            point = pose_like[0]
            if isinstance(point, (tuple, list, np.ndarray)) and len(point) >= 2:
                return float(point[0]), float(point[1])
        raise ValueError(f"Unsupported pose format for scene description: {type(pose_like)}")

    lines = ["Here is the scene description. x and y correspond to plane coordinates:", "Objects:"]
    for idx, loc in enumerate(world.locations):
        if loc is None:
            continue
        x, y = _xy_from_pose_like(loc)
        lines.append(
            f'tile{idx + 1}{{color:"green",location(x={x:.3f}, y={y:.3f}),id:{idx + 1}}}'
        )

    lines.append("")
    for idx, loc in enumerate(world.locations):
        if loc is None:
            continue
        x, y = _xy_from_pose_like(loc)
        lines.append(f"loc{idx + 1}{{x={x:.3f}, y={y:.3f}}}")

    lines.append("")
    lines.append("robot{location(x=0,y=0)}")
    template_path.write_text("\n".join(lines) + "\n")


def write_starter_file(num_locations: int) -> None:
    """Rewrite the `blocks`/`locations` reference sets in starter.txt to match
    the maximum number of block/location objects the generated domain declares
    (both sized to `grid_rows * grid_cols`)."""
    starter_path = pathlib.Path("envs/shelf/prompting/input/starter.txt")
    text = starter_path.read_text()

    blocks_str = ", ".join(f'"b{i}"' for i in range(1, num_locations + 1))
    locations_str = ", ".join(f'"loc{i}"' for i in range(1, num_locations + 1))

    text, n_blocks = re.subn(
        r'blocks = \{.*?\}', f"blocks = {{{blocks_str}}}", text, count=1
    )
    text, n_locations = re.subn(
        r'locations = \{.*?\}', f"locations = {{{locations_str}}}", text, count=1
    )
    if n_blocks != 1 or n_locations != 1:
        raise ValueError(f"Could not locate blocks/locations reference sets in {starter_path}")
    starter_path.write_text(text)


def add_grid_shelf(
    table: int,
    grid_rows: int,
    grid_cols: int,
    x_start: float,
    x_step: float,
    y_start: float,
    y_step: float,
    block_width: float,
    block_height: float,
) -> Tuple[List[int], float]:
    """Enclose the block grid on three sides with thin, transparent walls plus
    a lid on top, leaving only the near (minimum-y) side open. This visually
    and physically conveys that the robot can only grasp blocks by reaching
    in from the negative-Y direction (see `grasp_info` in `main`); the walls
    and lid are registered as fixed obstacles so motion planning keeps the
    robot from ever colliding with the shelf.

    Also returns `retract_y`: a world-frame y-coordinate, clear of the open
    side by another `margin_y`, that streams.py clamps IK approach poses to so
    the arm always fully retracts out from under the lid/walls before transit
    motion planning takes over (see `InverseKinematics` in envs/shelf/streams.py)."""
    margin_x = abs(x_step) * 1.3
    margin_y = abs(y_step) * 0.75
    tile_half_y = (block_width + 0.01) / 2  # matches the green tile marker's footprint
    x_min = min(x_start, x_start + (grid_cols - 1) * x_step) - margin_x
    x_max = max(x_start, x_start + (grid_cols - 1) * x_step) + margin_x
    # Open side: flush with the near edge of the closest tile row, not padded
    # like the other (walled) sides, so the shelf doesn't overhang the robot.
    y_min = min(y_start, y_start + (grid_rows - 1) * y_step) - tile_half_y
    y_max = max(y_start, y_start + (grid_rows - 1) * y_step) + margin_y  # walled (far) side

    table_top_z = pbt.utils.get_aabb(table).upper[2]
    wall_thickness = 0.01
    wall_height = block_height * 2
    wall_color = pbt.utils.RGBA(0.6, 0.8, 1.0, 0.25)
    wall_z = table_top_z + wall_height / 2
    lid_z = table_top_z + wall_height + wall_thickness / 2

    # Side walls run past the back wall's centerline to its outer face, and the
    # back wall is inset between their inner faces, so the corners are flush
    # (no gap or overlap) rather than mitered through each other's centerline.
    side_y_max = y_max + wall_thickness / 2
    side_center_y = (y_min + side_y_max) / 2

    shelf_bodies = []
    # Back wall (far side, max y), inset between the side walls.
    back = pbt.utils.create_box(x_max - x_min - wall_thickness, wall_thickness, wall_height, color=wall_color)
    pbt.utils.set_pose(back, pbt.utils.Pose(pbt.utils.Point(x=(x_min + x_max) / 2, y=y_max, z=wall_z)))
    shelf_bodies.append(back)
    # Left and right walls, extended to the back wall's outer face.
    for x_side in (x_min, x_max):
        side = pbt.utils.create_box(wall_thickness, side_y_max - y_min, wall_height, color=wall_color)
        pbt.utils.set_pose(side, pbt.utils.Pose(pbt.utils.Point(x=x_side, y=side_center_y, z=wall_z)))
        shelf_bodies.append(side)
    # Lid spanning the full grid footprint (including the open front), extended
    # by half the wall thickness on the walled sides so it sits flush with the
    # walls' outer faces rather than their centerlines.
    lid_x_min = x_min - wall_thickness / 2
    lid_x_max = x_max + wall_thickness / 2
    lid_y_max = y_max + wall_thickness / 2
    lid = pbt.utils.create_box(lid_x_max - lid_x_min, lid_y_max - y_min, wall_thickness, color=wall_color)
    pbt.utils.set_pose(
        lid, pbt.utils.Pose(pbt.utils.Point(x=(lid_x_min + lid_x_max) / 2, y=(y_min + lid_y_max) / 2, z=lid_z))
    )
    shelf_bodies.append(lid)
    # Clear not just the shelf, but a block sitting in the frontmost tile too:
    # the arm may be carrying another block back through that same tile's y,
    # and both blocks' footprints (not just the gripper's) need to pass clear
    # of each other, so add a full block_width of clearance on top of the
    # shelf-opening margin.
    retract_y = y_min - margin_y - block_width - 0.02
    return shelf_bodies, retract_y


def load_world(
    num_blocks: int = 5,
    timestep: bool = False,
    timestep_count: int = 20,
    grid_rows: int = 3,
    grid_cols: int = 3,
    goal_row: Optional[int] = None,
    goal_col: Optional[int] = None,
    debug: bool = False,
    shelf: bool = True,
) -> World:
    # TODO: store internal world info here to be reloaded
    pbt.utils.set_default_camera()
    # pbt.utils.draw_global_system()

    with pbt.utils.HideOutput():
        # add_data_path()
        models_dir = "models"
        robot = pbt.utils.load_model(
            pbt.utils.DRAKE_IIWA_URDF, fixed_base=True
        )  # DRAKE_IIWA_URDF | KUKA_IIWA_URDF PANDA_ARM_URDF
        table = pbt.utils.load_model(f"{models_dir}/long_floor.urdf")
        ab = dyn.urdf.load_model(
            str(
                pathlib.Path("external/pybullet-planning")
                / pbt.utils.DRAKE_IIWA_URDF
            )
        )
        block_width = 0.07
        block_height = 0.2

        red_id = num_blocks - 1
        blocks = []
        for i in range(num_blocks):
            if i != red_id:
                blocks.append(
                    pbt.utils.create_box(
                        block_width, block_width, block_height, color=pbt.utils.BLUE
                    )
                )
            else:
                blocks.append(
                    pbt.utils.create_box(
                        block_width, block_width, block_height, color=pbt.utils.RED
                    )
                )

    pbt.utils.draw_pose(
        pbt.utils.Pose(),
        parent=robot,
        parent_link=pbt.kuka_primitives.get_tool_link(robot),
    )  # TODO: not working
    num_locations = grid_rows * grid_cols
    # Location index = col * grid_rows + row (0-indexed), matching the
    # original 3x3 layout where x varies by column and y varies by row.
    # goal_row/goal_col are 1-indexed grid coordinates for the red block's
    # target location; default to the grid center.
    if goal_row is None:
        goal_row = grid_rows // 2 + 1
    if goal_col is None:
        goal_col = grid_cols // 2 + 1
    if not (1 <= goal_row <= grid_rows) or not (1 <= goal_col <= grid_cols):
        raise ValueError(
            f"goal_row/goal_col ({goal_row}, {goal_col}) out of bounds for a "
            f"{grid_rows}x{grid_cols} grid"
        )
    goal_idx = (goal_col - 1) * grid_rows + (goal_row - 1)
    goal_loc_name = f"loc{goal_idx + 1}"

    locations: List[Optional[pbt.utils.Pose]] = [None] * num_locations
    plocations: List[Optional[pbt.utils.Pose]] = [None] * num_locations
    z_block = pbt.utils.stable_z(blocks[0], table)

    x_start, x_step = -0.25, 0.2
    y_start, y_step = 0.9, -0.12
    for col in range(grid_cols):
        for row in range(grid_rows):
            idx = col * grid_rows + row
            x = x_start + col * x_step
            y = y_start + row * y_step
            locations[idx] = pbt.utils.Pose(pbt.utils.Point(x=x, y=y, z=z_block))

    new_box = pbt.utils.create_box(block_width + 0.01, block_width +
                                   0.01, 0.001, color=pbt.utils.GREEN)
    z_block = pbt.utils.stable_z(new_box, table)
    for col in range(grid_cols):
        for row in range(grid_rows):
            idx = col * grid_rows + row
            x = x_start + col * x_step
            y = y_start + row * y_step
            plocations[idx] = pbt.utils.Pose(pbt.utils.Point(x=x, y=y, z=z_block))

    if shelf:
        shelf_bodies, shelf_retract_y = add_grid_shelf(
            table, grid_rows, grid_cols, x_start, x_step, y_start, y_step, block_width, block_height
        )
    else:
        shelf_bodies, shelf_retract_y = [], None

    body_names = {
        table: "mytable",
    }
    shelf_part_names = ["shelf_back", "shelf_left", "shelf_right", "shelf_lid"]
    for shelf_body, shelf_name in zip(shelf_bodies, shelf_part_names):
        body_names[shelf_body] = shelf_name
    possible_locs = [i for i in range(num_locations) if i != goal_idx]
    new_state = set(["(handempty)"])
    loc_idxs = random.sample(possible_locs, num_blocks)
    line2 = f"(ontable b1 {goal_loc_name})"
    for idx in range(num_locations):
        new_box = pbt.utils.create_box(block_width + 0.01, block_width +
                                       0.01, 0.001, color=pbt.utils.GREEN)
        pbt.utils.set_pose(new_box, plocations[idx])
        if debug:
            pbt.utils.add_body_name(new_box, str(idx + 1), text_size=1)
    if num_blocks != 1:
        for idx in range(num_locations):
            if idx not in loc_idxs[1:] and idx != goal_idx:
                new_state.add(f"(clear loc{idx + 1})")
                line2 += f"(clear loc{idx + 1})"

        pbt.utils.set_pose(blocks[0], locations[goal_idx])  # b1 always occupies the goal location
        new_state.add(f"(ontable b1 {goal_loc_name})")
    else:
        for idx in range(num_locations):
            if idx != loc_idxs[0]:
                new_state.add(f"(clear loc{idx + 1})")
                line2 += f"(clear loc{idx + 1})"
    for i in range(1, num_blocks + 1):
        body_names[blocks[i - 1]] = f"b{i}"
        # new_state.add(f"(exist b{i})")
        # new_state.add(f"(clear b{i})")
        if i > 1 or num_blocks == 1:
            pbt.utils.set_pose(blocks[i - 1], locations[loc_idxs[i - 1]])
            # wait_for_user(f"blocks {i} {blocks[i - 1]} {loc_idxs[i - 1] + 1}")
            new_state.add(f"(ontable b{i} loc{loc_idxs[i - 1] + 1})")
            line2 += f"(ontable b{i} loc{loc_idxs[i - 1] + 1})"
            print(f"(ontable b{i} loc{loc_idxs[i - 1] + 1})", locations[loc_idxs[i - 1]])
    line2 += f"(handempty))(:goal (ontable b{num_blocks} {goal_loc_name})))"
    if timestep:
        for t in range(max(0, int(timestep_count) - 1)):
            new_state.add(f"(Next t{t + 1} t{t + 2})")
        new_state.add("(AtTimestep t1)")

    movable_bodies = blocks

    fixed_bodies = list(set(body_names.keys()) - set(movable_bodies))

    q_home = np.array([0.0, np.pi / 6, 0.0, -np.pi / 3, 0.0, np.pi / 2, 0.0])
    q0 = pbt.kuka_primitives.BodyConf(
        robot, pbt.utils.get_configuration(robot)
    )
    world = World(
        robot,
        ab,
        q_home,
        body_names,
        movable_bodies,
        fixed_bodies,
        table,
        block_height,
        new_state,
        locations,
        q0,
        grid_rows,
        grid_cols,
        goal_idx,
        shelf_bodies,
        shelf_retract_y,
    )
    write_scene_description(world, num_blocks)
    write_template_scene_description(world)
    write_starter_file(num_locations)
    return world


def init_geometric_state_kuka(
        num_blocks: int,
        timestep: bool,
        timestep_count: int,
        method: str,
        use_gui: bool = True,
        grid_rows: int = 3,
        grid_cols: int = 3,
        goal_row: Optional[int] = None,
        goal_col: Optional[int] = None,
        debug: bool = False,
        shelf: bool = True,
) -> Tuple[World, str, Set[str]]:
    pbt.utils.connect(use_gui=use_gui)
    world = load_world(
        num_blocks, timestep, timestep_count, grid_rows, grid_cols, goal_row, goal_col, debug, shelf
    )
    # prev_state_start = {}
    # prev_state_start["config"] = {
    #    "t1": (robot, BodyConf(robot, get_configuration(robot)))
    # }
    state = coast.GeometricState()
    objects: List[coast.Object] = [
        coast.Object(name=body_name, object_type="block", value=body_id)
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
            world.q0,
        )
    )

    for body_id in world.movable_bodies:
        body_pose = pbt.kuka_primitives.BodyPose(body_id, pbt.utils.get_pose(body_id))
        state.set(world.body_names[body_id], 0, (body_id, body_pose))

        stream_state.add(f"AtPose({world.body_names[body_id]}, p0_{body_id})")
        objects.append(coast.Object(f"p0_{body_id}", "pose", body_pose))
        # prev_state_start[world.body_names[obj]] = {
        #     "t1": (obj, pbt.kuka_primitives.BodyPose(obj, pbt.utils.get_pose(obj)))
        # }
        # prev_state_start[names[obj]] = {"t1": (obj, BodyPose(obj, get_pose(obj)))}
    for i, loc in enumerate(world.locations):
        objects.append(coast.Object(f"loc{i + 1}", "location", loc))
        state.set(f"loc{i + 1}", 0, loc)
    #     prev_state_start[f"loc{i + 1}"] = world.locations[i]
    #     prev_state_start[str(world.locations[i]).replace(" ", "")] = f"loc{i+1}"
    state.set("stacks", 0, {m: 1 for m in world.movable_bodies})

    problem = create_problem(world, num_blocks, method)
    return world, problem, state, stream_state, objects


def generate_domain_pddl(template_path: str, num_locations: int) -> str:
    """Rewrite a domain template's `b#`/`loc#` constants to cover `num_locations`
    blocks and locations, and write the result alongside the template."""
    text = pathlib.Path(template_path).read_text()
    constants = "\n".join(f"        b{i} - block" for i in range(1, num_locations + 1))
    constants += "\n" + "\n".join(f"        loc{i} - phys" for i in range(1, num_locations + 1))
    text, n_subs = re.subn(
        r"(\(:constants\n).*?(\n    \))",
        lambda m: m.group(1) + constants + m.group(2),
        text,
        count=1,
        flags=re.S,
    )
    if n_subs != 1:
        raise ValueError(f"Could not locate (:constants ...) block in {template_path}")
    out_path = str(pathlib.Path(template_path).with_name(
        pathlib.Path(template_path).stem + "_generated.pddl"
    ))
    pathlib.Path(out_path).write_text(text)
    return out_path


def update_goal(problem_pddl: str, num_blocks: int) -> None:
    # Change the goal to match the number of blocks
    with open(problem_pddl, "r") as f:
        problem_str = f.read()

    with open(problem_pddl, "w") as f:
        f.write(f"{problem_str[:problem_str.find('(:goal')]}")
        f.write(f"\n(:goal (ontable b{num_blocks} loc5))\n")
        f.write(")")


def create_problem(world: World, num_blocks: int, method: str) -> str:
    from symbolic import _P

    problem = symbolic.Problem(f"blocks-{num_blocks}", domain="blocks")
    for prop in world.new_state:
        problem.add_initial_prop(prop)

    problem.set_goal(_P("ontable", f"b{num_blocks}", f"loc{world.goal_loc + 1}"))
    # print("Printing problem:")
    # print(problem)

    return repr(problem)


def main(
        cfg: str,
        streams_py: str,
        algorithm: str,
        max_level: int,
        pybullet: bool,
        search_sample_ratio: float,
        timeout: int,
        reps: int,
        results_file: str,
        append: bool,
        random_seed: int,
        timestep: bool,
        use_gui: bool,
        skip_execution: bool,
        auto: bool,
        no_write: bool,
        continue_from: Optional[str] = None,
        grid_rows: int = 4,
        grid_cols: int = 3,
        goal_row: Optional[int] = None,
        goal_col: Optional[int] = None,
        debug: bool = False,
        skip_idx: Optional[int] = None,
        valid_seeds: Optional[Dict[int, List[int]]] = None,
        show_thinking: bool = False,
        no_shelf: bool = False,
):

    # Remove stale output files so the VLM is prompted fresh on the first experiment.
    # Oracles live in ../oracles/, outside this directory, so they are untouched.
    _output_dir = "envs/shelf/prompting/output"
    if os.path.isdir(_output_dir):
        for _fname in os.listdir(_output_dir):
            _fpath = os.path.join(_output_dir, _fname)
            if os.path.isfile(_fpath):
                os.remove(_fpath)

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
        print(f"[shelf.run] trial_mode output: {resolved_results_file}")

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
            f"[shelf.run] Continuing from {continue_from}, "
            f"skipping {rows_to_skip} completed experiment(s)"
        )

    experiment_counter = 0
    for cfg_idx, config in enumerate(experiments):
        if global_settings:
            for key, value in global_settings.items():
                if key not in config:
                    config[key] = value
                else:
                    print(f"[shelf.run] Warning: {key} already in config, skipping")

        for e in range(skip_idx or 0, reps):
            method = config.get('method', 'coast_z3')
            if method in ('coast_fd', 'coast_z3'):
                domain_pddl = "envs/shelf/domain_coast.pddl"
            elif method in ('zoast', 'VLMB', 'incremental'):
                domain_pddl = "envs/shelf/domain_z3.pddl"

            num_blocks = config['num_blocks']
            timestep_enabled = bool(config.get('timestep', timestep))
            timestep_count = int(config.get('timestep_count', 20))
            exp_grid_rows = int(config.get('grid_rows', grid_rows))
            exp_grid_cols = int(config.get('grid_cols', grid_cols))
            exp_goal_row = config.get('goal_row', goal_row)
            exp_goal_row = int(exp_goal_row) if exp_goal_row is not None else None
            exp_goal_col = config.get('goal_col', goal_col)
            exp_goal_col = int(exp_goal_col) if exp_goal_col is not None else None
            exp_shelf = bool(config.get('shelf', not no_shelf))
            domain_pddl = generate_domain_pddl(domain_pddl, exp_grid_rows * exp_grid_cols)

            # Determine the seed for this experiment. `valid_seeds` maps
            # num_blocks -> a list of pre-selected seeds that avoid impossible
            # problems for a given grid size, e.g. {"9": [0, 1, 3, 5, 6]} for a
            # 3x4 grid. It's specific to the grid size and rep count in use,
            # so it's config-driven rather than hardcoded.
            exp_valid_seeds = config.get('valid_seeds', valid_seeds or {})
            exp_valid_seeds = {int(k): v for k, v in exp_valid_seeds.items()}
            seed_e = exp_valid_seeds[num_blocks][e] if num_blocks in exp_valid_seeds else e

            # Print the VLM's chain-of-thought to the terminal (VLMB method only).
            config['show_thinking'] = config.get('show_thinking', show_thinking)

            # --continue-from: skip experiments already recorded in the CSV.
            if experiment_counter < rows_to_skip:
                experiment_counter += 1
                print(f"[shelf.run] Skipping experiment {experiment_counter}/{rows_to_skip} "
                      f"(cfg_idx={cfg_idx}, rep={e}, num_blocks={num_blocks})")
                continue
            experiment_counter += 1

            if random_seed is not None:
                np.random.seed(int(random_seed))
                random.seed(int(random_seed))
            else:
                np.random.seed(seed_e)
                random.seed(seed_e)

            (
                world,
                problem_pddl,
                state,
                stream_state,
                objects,
            ) = init_geometric_state_kuka(
                num_blocks,
                timestep_enabled,
                timestep_count,
                method,
                use_gui=use_gui,
                grid_rows=exp_grid_rows,
                grid_cols=exp_grid_cols,
                goal_row=exp_goal_row,
                goal_col=exp_goal_col,
                debug=debug,
                shelf=exp_shelf,
            )  ### Edit number of blocks here

            # update_goal(problem_pddl, num_blocks)
            # pbt.utils.wait_for_user()
            saver = pbt.utils.WorldSaver()
            pbt.utils.set_renderer(enable=True)

            # Extract parameters for image capture

            camera_pose = [-0.4, 0.1, 1] #[-0.4, -0.6, 1]
            target_pos = [-0.2, 0.5, 0] #perfect angles
            # image = get_image(camera_pos, target_pos, width=1024, height=768, vertical_fov=45.0)
            # Capture image using default camera settings
            image = pbt.utils.get_image(camera_pose, target_pos,width=1024, height=768,vertical_fov=45.0)
            # Save image
            pbt.utils.save_image("envs/shelf/prompting/input/default_camera_capture.png", image.rgbPixels)

            if timestep_enabled:
                constraint_to_use = Constraint
                # use_cache = True
            else:
                constraint_to_use = LocationConstraint
                print("[shelf.run] constraint used LocationConstraint")
                use_cache = False

            if not auto:
                pbt.utils.wait_for_user("Begin planning?")

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
                constraint_cls=constraint_to_use,
                use_cache=False,
                sort_streams=True
            )
            print("[shelf.run] PLANNING DONE")

            res = plan.log.get_timing()
            res['success'] = True
            res['rep_idx'] = seed_e
            res['num_blocks'] = config['num_blocks']
            if (plan.action_plan is None or plan.objects is None):
                print("[shelf.run] No plan found")
                res['success'] = False
                
            if not no_write:
                file_mode = "a" if csv_initialized or append else "w"
                # Lock column order on first write so that later rows
                # (whose dict key order may differ) stay aligned with
                # the header.
                if csv_fieldnames is None:
                    csv_fieldnames = list(res.keys())
                with open(resolved_results_file, file_mode, newline="") as f_object:
                    writer_csv = csv.DictWriter(f_object, fieldnames=csv_fieldnames)
                    if not csv_initialized:
                        writer_csv.writeheader()
                        csv_initialized = True
                    writer_csv.writerow(res)

            if res['success']:
                pbt.utils.set_renderer(enable=True)
                saver.restore()
                plan.log.print()
                if pybullet:
                    if not auto:
                        pbt.utils.wait_for_user("Execute?")
                    if not skip_execution:
                        a = 1.2
                        time.sleep(1)
                        for action in plan.action_plan:
                            # text = pbt.utils.add_text(action, (0.45, -0.2, a))
                            action.execute(plan.objects)
                            # pbt.utils.remove_debug(text)
                            # breakpoint()
                        # pbt.utils.wait_for_user("Finish?")
            time.sleep(1)
            if pbt.utils.is_connected():
                pbt.utils.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--timestep",
        type=bool,
        default=False)
    parser.add_argument(
        "--streams-py",
        nargs="?",
        default="envs/shelf/streams.py",
        help="Python streams file.",
    )
    parser.add_argument(
        "--algorithm",
        nargs="?",
        default="improved",
        help="Max planning level.",
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
        default=1.0,  # 1.0 = don't try to resample.
        help="The ratio of search versus sample",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout for algorithm"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=0,
        help="The number of times you want to run the experiment",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="The file to write the csv of experiment",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the results file",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="envs/shelf/config.json",
        help="configuration file"
    )

    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-write", action="store_true", help="Skip writing to the results file")
    parser.add_argument("--use-gui", action="store_true", help="Use GUI")
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip executing the plan on the robot. Does not affect the "
                             "'Begin planning?'/'Execute?' prompts; use --auto for that.")
    parser.add_argument("--auto", action="store_true",
                        help="Skip the 'Begin planning?' and 'Execute?' confirmation prompts "
                             "and run straight through.")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Resume from an existing CSV, appending new results and skipping completed experiments")
    parser.add_argument("--debug", action="store_true",
                        help="Label each location tile with its PDDL name (loc1, loc2, ...) in the GUI.")
    parser.add_argument("--skip-idx", type=int, default=None,
                        help="Start from the nth repetition of the experiment (0-indexed)")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Print the VLM's chain-of-thought to the terminal (VLMB method only).")
    parser.add_argument("--no-shelf", action="store_true",
                        help="Disable the transparent shelf enclosure around the block grid. "
                             "Overridable per-experiment via config's 'shelf'.")

    args = parser.parse_args()
    print(args)
    try:
        main(**vars(args))
    except Exception as e:
        if pbt.utils.is_connected():
            pbt.utils.disconnect()
        raise e
