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

import numpy as np
import pybullet_tools as pbt
from pybullet_tools import kuka_primitives, utils as pb_utils
import spatialdyn as dyn

import coast
from coast.constraint import Constraint


# --- Scene layout ----------------------------------------------------------
# long_floor.urdf is a 2 (x) by 4 (y) slab whose collision box sits at
# z in [-0.06, 0.04], so the table surface is at z = 0.04.
TABLE_TOP_Z = 0.04

# The two arm bases are 1.0m apart along y.  Every own-side target is within
# 0.70m of its arm (packing's demonstrated working envelope) and every
# cross-side target is at least 1.09m away, well past the iiwa's ~0.82m
# structural limit for a top grasp.  That gap is what makes "an arm cannot
# reach the other side" fail during refinement rather than sometimes succeed.
ARM_BASES = {
    "left": (0.00, +0.50),
    "right": (0.00, -0.50),
}
POT_XY = {
    "frypot": (0.00, +0.98),    # LEFT side: frying
    "steampot": (0.00, -0.98),  # RIGHT side: steaming
}
PLATE_XY = {
    # Side plates sit on the camera's side of the arms (negative x) so nothing
    # occludes them in the rendered view.  Reflecting them across the y axis
    # leaves every arm-to-target distance unchanged, because both arm bases are
    # at x = 0 -- the reach budget below is unaffected.
    "plateleft": (-0.40, +0.70),
    "plateright": (-0.40, -0.70),
    "platemid": (0.15, 0.00),
}

# Ingredients are 0.06 x 0.06 x 0.05 (packing's sblock, sized to roughly match
# the end-effector wrist flange).  Four fit on a plate in a 2x2 grid.
INGREDIENT_DIMS = (0.06, 0.06, 0.05)
INGREDIENT_GRID = [(-0.065, -0.065), (-0.065, 0.065), (0.065, -0.065), (0.065, 0.065)]
MAX_PER_PLATE = len(INGREDIENT_GRID)

# An ingredient's HUE encodes its goal status and its SHADE encodes whether it
# has got there yet: a fried-goal ingredient starts light red and darkens to
# deep red the moment it is actually fried, likewise light -> dark blue for
# steaming.  An uncooked-goal ingredient has nothing to reach, so it is green
# throughout.  Both tables are keyed by goal status; streams.py darkens a body
# only when the cooking it undergoes matches the goal, so frying something
# destined for the steamer leaves it light blue -- visibly not done.
STATUS_COLORS = {
    "uncooked": pb_utils.RGBA(0.25, 0.70, 0.30, 1),
    "fried": pb_utils.RGBA(1.00, 0.60, 0.55, 1),
    "steamed": pb_utils.RGBA(0.62, 0.78, 1.00, 1),
}
COOKED_COLORS = {
    "uncooked": STATUS_COLORS["uncooked"],
    "fried": pb_utils.RGBA(0.50, 0.04, 0.03, 1),
    "steamed": pb_utils.RGBA(0.05, 0.10, 0.60, 1),
}

# Both pots come from one URDF, so their colours are applied per body after
# loading.  Alpha is per link: the base slab is opaque and the four walls stay
# translucent, so the camera still sees what is inside.
POT_COLORS = {
    "frypot": (0.95, 0.45, 0.05),    # orange
    "steampot": (0.00, 0.75, 0.85),  # cyan
}
POT_WALL_ALPHA = 0.35

GOAL_PLATES = ("plateleft", "plateright", "platemid")
GOAL_STATUSES = ("uncooked", "fried", "steamed")


def _color_pot(body: int, rgb: Tuple[float, float, float]) -> None:
    for link in pb_utils.get_all_links(body):
        alpha = 1.0 if link == pb_utils.BASE_LINK else POT_WALL_ALPHA
        pb_utils.set_color(body, pb_utils.RGBA(*rgb, alpha), link=link)


@dataclasses.dataclass
class Arm:
    name: str
    """PDDL constant for this arm ("left" or "right")."""

    robot: int
    """PyBullet body id; also the value of the coast.Object named `name`."""

    ab: dyn.ArticulatedBody
    """Own spatialdyn model.  compute_opspace_ik mutates .q as the IK seed and
    the approach->grasp solves chain seeds, so the arms cannot share one."""

    base_pose: Any
    inv_base_pose: Any
    """Cached pb_utils.invert(base_pose).  `ab` is built with its base at the
    identity, so world poses must be pushed through this before IK."""

    tool_link: int
    q_home: np.ndarray
    q_park: np.ndarray
    fixed_bodies: List[int]
    """Motion-planner obstacles: the scene's fixed bodies plus the OTHER arm."""


@dataclasses.dataclass
class World:
    arms: Dict[str, Arm]
    arms_by_id: Dict[int, Arm]
    """robot body id -> Arm.  The streams' dispatch table for the ?a input."""

    body_names: Dict[int, str]
    body_types: Dict[int, str]
    """ingredient | lid | pot | plate | static."""

    movable_bodies: List[int]
    fixed_bodies: List[int]
    """Scene fixed bodies only -- no robots.  See Arm.fixed_bodies."""

    new_state: Set[str]
    goal_assignment: Dict[str, Tuple[str, str]]
    """ingredient name -> (target plate, target status)."""

    fryer: int
    steamer: int
    """Body ids of the pots that (isfryer)/(issteamer) name in :init.  An
    action's execute() only receives its geometric inputs/outputs -- never the
    ?on parameter -- so the streams recover the target pot from where the body
    landed and need to tell the two apart."""

    cooked_colors: Dict[str, Any]
    """goal status -> the darkened colour an ingredient takes on reaching it."""


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
        size_arr = np.array(_parse_xyz_attr(box.attrib.get("size")), dtype=float)

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

    if wall_x_thickness is None:
        wall_x_thickness = 0.01
    if wall_y_thickness is None:
        wall_y_thickness = 0.01

    internal_x = max(0.0, base_size[0] - 2.0 * wall_x_thickness)
    internal_y = max(0.0, base_size[1] - 2.0 * wall_y_thickness)
    base_top_z = base_origin[2] + base_size[2] / 2.0
    internal_z = 0.0 if wall_top_z is None else max(0.0, wall_top_z - base_top_z)

    return base_size, np.array([internal_x, internal_y, internal_z], dtype=float)


def _ingredient_names(num_ingredients: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """Split ing1..ingN between the left and right starting plates.

    num_ingredients is {"left": n, "right": n}.  Names are handed out left
    plate first, so ing1..ing<left> start on plateleft and the rest on
    plateright.
    """
    if not isinstance(num_ingredients, dict):
        raise ValueError(
            f'num_ingredients must be {{"left": n, "right": n}}, got {num_ingredients!r}'
        )
    unknown = set(num_ingredients) - {"left", "right"}
    if unknown:
        raise ValueError(
            f"unknown num_ingredients key(s) {sorted(unknown)}; expected 'left' and 'right'"
        )
    n_left = int(num_ingredients.get("left", 0))
    n_right = int(num_ingredients.get("right", 0))
    if n_left > MAX_PER_PLATE or n_right > MAX_PER_PLATE:
        raise ValueError(
            f"at most {MAX_PER_PLATE} ingredients fit on a plate, got {num_ingredients}"
        )
    if n_left < 0 or n_right < 0 or n_left + n_right == 0:
        raise ValueError(f"need at least one ingredient, got {num_ingredients}")
    names = [f"ing{i + 1}" for i in range(n_left + n_right)]
    return names[:n_left], names[n_left:]


def _generate_goal_assignment(
    names: List[str], start_plate: Dict[str, str], goal_seed: int
) -> Dict[str, Tuple[str, str]]:
    """Assign each ingredient a target plate and a target cooking status.

    Uses its own Random so the global RNG (which drives the geometric samplers)
    stays reproducible independent of the goal draw.  Re-draws assignments that
    the initial state already satisfies, so every instance needs real work.
    """
    rng = random.Random(goal_seed)
    assignment: Dict[str, Tuple[str, str]] = {}
    for name in names:
        while True:
            plate = rng.choice(GOAL_PLATES)
            status = rng.choice(GOAL_STATUSES)
            # Ingredients start uncooked on their own side's plate.
            if status == "uncooked" and plate == start_plate[name]:
                continue
            assignment[name] = (plate, status)
            break
    return assignment


def _parse_goal_assignment(
    goals: Any, names: List[str]
) -> Dict[str, Tuple[str, str]]:
    """Validate an explicit config goal list into {ingredient: (plate, status)}.

    Expects `[[name, plate, status], ...]` covering every ingredient in the
    instance exactly once.  Errors here are config typos, so they are raised
    loudly rather than silently defaulted -- a mis-typed plate name would
    otherwise become an unreachable goal that looks like a planner failure.
    """
    if not isinstance(goals, (list, tuple)):
        raise ValueError(f"'goals' must be a list of [name, plate, status], got {type(goals).__name__}")

    assignment: Dict[str, Tuple[str, str]] = {}
    for entry in goals:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(f"goal entry must be [name, plate, status], got {entry!r}")
        name, plate, status = (str(x) for x in entry)
        if name not in names:
            raise ValueError(
                f"goal names ingredient {name!r}, but this instance only has {names}. "
                f"num_ingredients determines how many exist."
            )
        if name in assignment:
            raise ValueError(f"ingredient {name!r} is assigned a goal twice")
        if plate not in GOAL_PLATES:
            raise ValueError(f"unknown goal plate {plate!r}; expected one of {list(GOAL_PLATES)}")
        if status not in GOAL_STATUSES:
            raise ValueError(f"unknown goal status {status!r}; expected one of {list(GOAL_STATUSES)}")
        assignment[name] = (plate, status)

    missing = [n for n in names if n not in assignment]
    if missing:
        raise ValueError(f"no goal given for {missing}; every ingredient needs one")
    return assignment


def write_scene_description(world: World) -> None:
    scene_path = pathlib.Path("envs/cooking/prompting/input/scene_description.txt")
    scene_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Here is the scene description. x and y correspond to plane coordinates:",
        "Objects:",
    ]
    pot_urdf = pathlib.Path("models/cooking_pot.urdf")
    pot_internal_dims = None
    if pot_urdf.exists():
        _, pot_internal_dims = _parse_urdf_container_dims(str(pot_urdf))

    for body_id, name in world.body_names.items():
        pddl_type = world.body_types[body_id]
        if pddl_type == "static":
            continue
        point, _ = pb_utils.get_pose(body_id)
        aabb = pb_utils.get_aabb(body_id)
        ext = np.array(aabb.upper) - np.array(aabb.lower)
        dims_field = f"external_dims(x={ext[0]:.3f}, y={ext[1]:.3f}, z={ext[2]:.3f})"
        if pddl_type == "pot" and pot_internal_dims is not None:
            dims_field += (
                f", internal_dims(x={pot_internal_dims[0]:.3f}, "
                f"y={pot_internal_dims[1]:.3f}, z={pot_internal_dims[2]:.3f})"
            )
        lines.append(
            f'{name}{{type:"{pddl_type}",color:"{_scene_color_for_body(world, body_id)}",'
            f"location(x={point[0]:.3f}, y={point[1]:.3f}),{dims_field}}}"
        )
    lines.append("")
    for arm in world.arms.values():
        point = arm.base_pose[0]
        lines.append(f"{arm.name}{{type:\"arm\",location(x={point[0]:.3f}, y={point[1]:.3f})}}")
    scene_path.write_text("\n".join(lines) + "\n")


def _scene_color_for_body(world: World, body_id: int) -> str:
    name = world.body_names[body_id]
    pddl_type = world.body_types[body_id]
    if pddl_type == "ingredient":
        # The description is written before anything is cooked, so these are
        # the pre-cooking shades; each darkens once the ingredient is cooked.
        return {"uncooked": "green", "fried": "light red", "steamed": "light blue"}[
            world.goal_assignment[name][1]
        ]
    if pddl_type == "lid":
        return "grey"
    if pddl_type == "pot":
        return {"frypot": "transparent orange", "steampot": "transparent cyan"}[name]
    if pddl_type == "plate":
        return "white"
    return "gray"


def _load_arm(name: str, models_dir: str, q_home: np.ndarray) -> Arm:
    x, y = ARM_BASES[name]
    base_pose = pb_utils.Pose(pb_utils.Point(x, y, 0.0))
    robot = pbt.utils.load_model(
        pbt.utils.DRAKE_IIWA_URDF, pose=base_pose, fixed_base=True
    )
    # A separate spatialdyn model per arm: its base sits at the identity, which
    # is why solve_arm_ik() in streams.py transforms world poses by
    # inv_base_pose before handing them over.
    ab = dyn.urdf.load_model(
        str(pathlib.Path("external/pybullet-planning/") / pbt.utils.DRAKE_IIWA_URDF)
    )
    return Arm(
        name=name,
        robot=robot,
        ab=ab,
        base_pose=base_pose,
        inv_base_pose=pb_utils.invert(base_pose),
        tool_link=kuka_primitives.get_tool_link(robot),
        q_home=q_home,
        q_park=q_home,
        fixed_bodies=[],
    )


# num_ingredients is {"left": n, "right": n} and fixes the INITIAL state only.
# The goal comes from `goals` when the config supplies one, otherwise it is
# drawn randomly using goal_seed.
def load_world(
    num_ingredients: Dict[str, int], goal_seed: int = 0, goals: Optional[Any] = None
) -> World:
    pbt.utils.set_default_camera()

    cwd = os.getcwd()
    print(f"cwd: {cwd}")
    q_home = np.array([0.0, -np.pi / 4, 0.0, -np.pi / 2, 0.0, np.pi / 3, 0.0])

    with pbt.utils.HideOutput():
        models_dir = f"{cwd}/models"

        table = pbt.utils.load_model(f"{models_dir}/long_floor.urdf")

        arms = {name: _load_arm(name, models_dir, q_home) for name in ("left", "right")}

        body_names: Dict[int, str] = {table: "table"}
        body_types: Dict[int, str] = {table: "static"}

        # Every z below comes from stable_z rather than a literal.  PyBullet
        # re-frames a URDF body onto its inertial origin, and all three of
        # these models declare a non-zero one, so a hand-computed z is off by
        # 0.01-0.025 depending on the model.
        pots: Dict[str, int] = {}
        for pot_name, (x, y) in POT_XY.items():
            body = pbt.utils.load_model(
                f"{models_dir}/cooking_pot.urdf",
                pose=pb_utils.Pose(pb_utils.Point(x, y, 0.0)),
            )
            pb_utils.set_pose(
                body,
                pb_utils.Pose(pb_utils.Point(x, y, pb_utils.stable_z(body, table))),
            )
            _color_pot(body, POT_COLORS[pot_name])
            pots[pot_name] = body
            body_names[body] = pot_name
            body_types[body] = "pot"

        plates: Dict[str, int] = {}
        for plate_name, (x, y) in PLATE_XY.items():
            body = pbt.utils.load_model(
                f"{models_dir}/plate.urdf",
                pose=pb_utils.Pose(pb_utils.Point(x, y, 0.0)),
            )
            pb_utils.set_pose(
                body,
                pb_utils.Pose(pb_utils.Point(x, y, pb_utils.stable_z(body, table))),
            )
            plates[plate_name] = body
            body_names[body] = plate_name
            body_types[body] = "plate"

        lids: Dict[str, int] = {}
        for lid_name, pot_name in (("lidleft", "frypot"), ("lidright", "steampot")):
            x, y = POT_XY[pot_name]
            body = pbt.utils.load_model(
                f"{models_dir}/cooking_lid.urdf",
                pose=pb_utils.Pose(pb_utils.Point(x, y, 0.0)),
            )
            pb_utils.set_pose(
                body,
                pb_utils.Pose(
                    pb_utils.Point(x, y, pb_utils.stable_z(body, pots[pot_name]))
                ),
            )
            lids[lid_name] = body
            body_names[body] = lid_name
            body_types[body] = "lid"

        left_names, right_names = _ingredient_names(num_ingredients)
        start_plate = {n: "plateleft" for n in left_names}
        start_plate.update({n: "plateright" for n in right_names})
        all_names = left_names + right_names
        if goals is not None:
            goal_assignment = _parse_goal_assignment(goals, all_names)
        else:
            goal_assignment = _generate_goal_assignment(all_names, start_plate, goal_seed)

        ingredients: Dict[str, int] = {}
        w1, w2, h = INGREDIENT_DIMS
        for names, plate_name in ((left_names, "plateleft"), (right_names, "plateright")):
            px, py = PLATE_XY[plate_name]
            for i, name in enumerate(names):
                # Colour by the ingredient's GOAL status so the camera image
                # the VLM sees encodes the goal, not just the layout.
                color = STATUS_COLORS[goal_assignment[name][1]]
                body = pbt.utils.create_box(w1, w2, h, color=color)
                dx, dy = INGREDIENT_GRID[i]
                x, y = px + dx, py + dy
                pb_utils.set_pose(body, pb_utils.Pose(pb_utils.Point(x, y, 0.0)))
                pb_utils.set_pose(
                    body,
                    pb_utils.Pose(
                        pb_utils.Point(
                            x, y, pb_utils.stable_z(body, plates[plate_name])
                        )
                    ),
                )
                ingredients[name] = body
                body_names[body] = name
                body_types[body] = "ingredient"

    for arm in arms.values():
        pb_utils.draw_pose(pb_utils.Pose(), parent=arm.robot, parent_link=arm.tool_link)

    fixed_bodies = [table] + list(pots.values()) + list(plates.values())
    movable_bodies = list(ingredients.values()) + list(lids.values())

    # Each arm plans against the scene AND the other arm.
    for arm in arms.values():
        arm.fixed_bodies = fixed_bodies + [
            other.robot for other in arms.values() if other.robot != arm.robot
        ]

    # Build PDDL initial state.
    new_state: Set[str] = set()
    for name in ingredients:
        new_state.add(f"(on {name} {start_plate[name]})")
        new_state.add(f"(uncooked {name})")
    new_state.add("(on lidleft frypot)")
    new_state.add("(on lidright steampot)")
    new_state.add("(isfryer frypot)")
    new_state.add("(issteamer steampot)")
    # Anything an item may be set down on.  Not the table (nothing is ever
    # placed there), not the arms, and not the ing1..ing8 constants -- without
    # this the planner happily places onto an arm, or onto an ingredient
    # constant that this instance never instantiated and which therefore has no
    # geometric counterpart for the streams to bind.
    for name in list(POT_XY) + list(PLATE_XY) + ["lidleft", "lidright"]:
        new_state.add(f"(support {name})")

    world = World(
        arms=arms,
        arms_by_id={arm.robot: arm for arm in arms.values()},
        body_names=body_names,
        body_types=body_types,
        movable_bodies=movable_bodies,
        fixed_bodies=fixed_bodies,
        new_state=new_state,
        goal_assignment=goal_assignment,
        fryer=pots["frypot"],
        steamer=pots["steampot"],
        cooked_colors=COOKED_COLORS,
    )
    write_scene_description(world)
    return world


def init_geometric_state_cooking(
    num_ingredients: Dict[str, int], goal_seed: int = 0, use_gui: bool = True,
    goals: Optional[Any] = None,
) -> Tuple[World, Any, Set[str], List[Any]]:
    pbt.utils.connect(use_gui=use_gui)
    world = load_world(num_ingredients, goal_seed=goal_seed, goals=goals)

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

    # The arms have to exist as coast Objects: retrace() resolves the PDDL
    # constant `left` through this map to hand the streams the robot's body id.
    # Their "arm" type also keeps them out of the ingredient/lid _foralls.
    for arm in world.arms.values():
        objects.append(coast.Object(arm.name, "arm", arm.robot))
        conf_name = f"q0_{arm.name}"
        conf = pbt.kuka_primitives.BodyConf(
            arm.robot, pbt.utils.get_configuration(arm.robot)
        )
        state.set("config", 0, (arm.robot, conf))
        stream_state.add(f"AtConf({arm.name}, {conf_name})")
        objects.append(coast.Object(conf_name, "conf", conf))

    state.set("cur_grasp", 0, None)
    state.set("commands", 0, [])

    for body_id in world.movable_bodies:
        body_pose = pbt.kuka_primitives.BodyPose(body_id, pbt.utils.get_pose(body_id))
        state.set(world.body_names[body_id], 0, (body_id, body_pose))
        stream_state.add(f"AtPose({world.body_names[body_id]}, p0_{body_id})")
        objects.append(coast.Object(f"p0_{body_id}", "pose", body_pose))

    return world, state, stream_state, objects


# Every scene object is a domain constant, so the problem's :objects block is
# empty (the timesteps live in the domain's :constants).
DOMAIN_CONSTANTS = {
    "table",
    "left",
    "right",
    "frypot",
    "steampot",
    "plateleft",
    "plateright",
    "platemid",
    "lidleft",
    "lidright",
} | {f"ing{i + 1}" for i in range(8)}


def create_problem(
    world: World,
    dynamic_timesteps: bool = False,
    timestep_count: Optional[int] = None,
    include_timesteps: bool = True,
) -> str:
    """Build the PDDL problem string and generate temp_domain.pddl with timesteps.

    Returns the problem PDDL string (also written to envs/cooking/temp_problem.pddl).
    """
    num_ingredients = sum(
        1 for bt in world.body_types.values() if bt == "ingredient"
    )
    if timestep_count is None:
        if dynamic_timesteps:
            # Per ingredient: pick+place into a pot to cook it, then
            # pick+place onto its goal plate.  Plus lid removal and
            # replacement on both pots.
            timestep_count = num_ingredients * 4 + 8
        else:
            timestep_count = 30
    timestep_count = max(2, int(timestep_count))

    objects_lines = [
        f"        {name} - {world.body_types[body_id]}"
        for body_id, name in world.body_names.items()
        if name not in DOMAIN_CONSTANTS
    ]
    objects_str = "\n".join(objects_lines)

    init_lines = [f"        {prop}" for prop in sorted(world.new_state)]
    if include_timesteps:
        init_lines.append("        (AtTimestep t1)")
        for t in range(timestep_count - 1):
            init_lines.append(f"        (Next t{t + 1} t{t + 2})")
    init_str = "\n".join(init_lines)

    goal_parts = []
    for name in sorted(world.goal_assignment, key=lambda n: int(n[3:])):
        plate, status = world.goal_assignment[name]
        goal_parts.append(f"(on {name} {plate})")
        goal_parts.append(f"({status} {name})")
    goal = "(and\n" + "\n".join(f"        {g}" for g in goal_parts) + "\n    )"

    problem_pddl = f"""(define (problem cooking-task)
    (:domain cooking)
    (:objects
{objects_str}
    )
    (:init
{init_str}
    )
    (:goal {goal})
)
"""

    with open("envs/cooking/temp_problem.pddl", "w") as f:
        f.write(problem_pddl)

    if include_timesteps:
        with open("envs/cooking/domain_coast.pddl", "r") as f:
            domain_str = f.read()

        timesteps_insertion = ""
        for i in range(timestep_count):
            timesteps_insertion += f"\n        t{i + 1} - timestep"
        domain_str = domain_str.replace(";timesteps here", timesteps_insertion)

        with open("envs/cooking/temp_domain.pddl", "w") as f:
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

    # Remove stale output files so the VLM is prompted fresh on the first
    # experiment.  Oracles live in ../oracles/, so they are untouched.
    _output_dir = "envs/cooking/prompting/output"
    if os.path.isdir(_output_dir):
        for _fname in os.listdir(_output_dir):
            _fpath = os.path.join(_output_dir, _fname)
            if os.path.isfile(_fpath):
                os.remove(_fpath)

    experiments = json.load(open(cfg))

    if type(experiments) == dict:
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
        print(f"[cooking.run] trial_mode output: {resolved_results_file}")

    csv_initialized = False
    csv_fieldnames = None  # locked on first write to keep column order stable

    rows_to_skip = 0
    if continue_from is not None:
        if not os.path.exists(continue_from):
            raise FileNotFoundError(f"--continue-from file not found: {continue_from}")
        resolved_results_file = continue_from
        append = True
        csv_initialized = True
        with open(continue_from, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                csv_fieldnames = header
            rows_to_skip = sum(1 for _ in reader)
        print(
            f"[cooking.run] Continuing from {continue_from}, "
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

        num_ingredients = config['num_ingredients']

        if config.get("skip", False):
            continue

        for e in range(reps):
            if experiment_counter < rows_to_skip:
                experiment_counter += 1
                print(f"[cooking.run] Skipping experiment {experiment_counter}/{rows_to_skip} "
                      f"(cfg_idx={cfg_idx}, rep={e}, num_ingredients={num_ingredients})")
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
            ) = init_geometric_state_cooking(
                num_ingredients, goal_seed=cfg_idx, use_gui=use_gui,
                goals=config.get("goals"),
            )

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
            if use_timesteps:
                # coast_fd and coast_z3 share the same domain (same :adl
                # requirements and type hierarchy).
                domain_pddl = "envs/cooking/temp_domain.pddl"
            else:
                domain_pddl = "envs/cooking/domain_z3.pddl"

            config['show_thinking'] = config.get('show_thinking', show_thinking)

            saver = pbt.utils.WorldSaver()
            pbt.utils.set_renderer(enable=True)
            # Head-on from -x, so that the LEFT half of the scene (plateleft,
            # frypot) lands on the left of the image and the right half on the
            # right.  Viewed from +x the two would be mirrored, which is
            # actively misleading for a VLM being asked which arm can reach
            # what.
            camera_pose = [-2.1, 0.0, 1.1]
            target_pos = [-0.10, 0.0, 0.35]
            image = pbt.utils.get_image(
                camera_pose, target_pos, width=1024, height=768, vertical_fov=45.0
            )
            pbt.utils.save_image(
                "envs/cooking/prompting/input/default_camera_capture.png", image.rgbPixels
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

            res = plan.log.get_timing()
            res['success'] = True
            res['rep_idx'] = e
            res['num_ingredients'] = num_ingredients
            if plan.action_plan is None or plan.objects is None:
                print("No plan found")
                res['success'] = False

            if not no_write:
                file_mode = "a" if csv_initialized or append else "w"
                if csv_fieldnames is None:
                    csv_fieldnames = list(res.keys())
                with open(resolved_results_file, file_mode, newline="") as f_object:
                    writer_csv = csv.DictWriter(f_object, fieldnames=csv_fieldnames)
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
        default="envs/cooking/streams.py",
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
        default="envs/cooking/debug.json",
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
