#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import os
import pathlib
import random
import time
import json
from typing import Dict, List, Optional, Set, Tuple, Any
import sys

import numpy as np
import pybullet_tools as pbt
from pybullet_tools import kuka_primitives, utils as pb_utils
import spatialdyn as dyn
import symbolic

import coast
from coast.constraint import Constraint

# from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF

@dataclasses.dataclass
class World:
    robot: int
    ab: dyn.ArticulatedBody
    body_names: Dict[int, str]
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


def _kuka_type_for_name(name: str) -> str:
    if name.startswith("lid"):
        return "lid"
    if name in {"egg", "bacon", "celery", "radish", "g1", "g2"}:
        return "item"
    if name in {"mysink", "mystove", "mynorth", "myplate", "mytable"}:
        return "surface"
    return "object"


def _kuka_color_for_name(name: str) -> str:
    if name in {"mysink", "lid1", "lid2", "lid3", "celery", "radish"}:
        return "blue"
    if name in {"mystove", "egg", "bacon"}:
        return "red"
    if name in {"mynorth", "g1", "g2"}:
        return "green"
    if name == "myplate":
        return "yellow"
    return "gray"


def write_scene_description(world: World) -> None:
    scene_path = pathlib.Path("envs/containers/prompting/input/scene_description.txt")
    scene_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["Here is the scene description. x and y correspond to plane coordinates:", "Objects:"]
    for body_id, name in world.body_names.items():
        point, _ = pb_utils.get_pose(body_id)
        aabb = pb_utils.get_aabb(body_id)
        ext = np.array(aabb.upper) - np.array(aabb.lower)
        lines.append(
            f'{name}{{type:"{_kuka_type_for_name(name)}",color:"{_kuka_color_for_name(name)}",'
            f'location(x={point[0]:.3f}, y={point[1]:.3f}),'
            f'external_dims(x={ext[0]:.3f}, y={ext[1]:.3f}, z={ext[2]:.3f})}}'
        )
    lines.append("")
    lines.append("robot{location(x=0,y=0)}")
    scene_path.write_text("\n".join(lines) + "\n")


def load_world() -> World:
    # TODO: store internal world info here to be reloaded
    pbt.utils.set_default_camera()
    # pbt.utils.draw_global_system()

    with pbt.utils.HideOutput():
        models_dir = "../../models"


        # robot = pbt.utils.load_model(
        #     pbt.utils.PANDA_ARM_URDF, fixed_base=True
        # )  # DRAKE_IIWA_URDF | KUKA_IIWA_URDF | PANDA_ARM_URDF DRAKE_IIWA_URDF PANDA_TEST_URDF PANDA_URDF
        table = pbt.utils.load_model(f"{models_dir}/long_floor.urdf")
        robot = pbt.utils.load_model(pbt.utils.DRAKE_IIWA_URDF, pose=pb_utils.Pose(pb_utils.Point(y=+0.0)))
        
        ab = dyn.urdf.load_model(
            str(
                pathlib.Path("external/pybullet-planning/")
                / pbt.utils.DRAKE_IIWA_URDF
            )
        )
        blue_box=[-0.0,-0.4]
        red_box = [ 0.0, 0.4]
        green_box = [0.45, 0]
        yellow_box = [0.3,0.5]

        north_path = f"{models_dir}/north.urdf"
        south_path = f"{models_dir}/south.urdf"
        g1_path = f"{models_dir}/g1.urdf"
        g2_path = f"{models_dir}/g2.urdf"
        sink_path = f"{models_dir}/sink.urdf"
        stove_path = f"{models_dir}/stove.urdf"
        sink = pb_utils.load_model(
            sink_path, pose=pb_utils.Pose(pb_utils.Point(blue_box[1],blue_box[0]))
        )
        stove = pb_utils.load_model(
            stove_path, pose=pb_utils.Pose(pb_utils.Point(red_box[1],red_box[0]))
        )
        north = pb_utils.load_model(
            north_path, pose=pb_utils.Pose(pb_utils.Point(green_box[1],green_box[0]))
        )
        # south = pb_utils.load_model(
        #     south_path, pose=pb_utils.Pose(pb_utils.Point(y=-0.3,x=-0.3))
        # )

        container_path1= f"{models_dir}/container_lid.urdf"
        container_path2 = f"{models_dir}/container_lid2.urdf"
        container_path3 = f"{models_dir}/container_lid3.urdf"
        container_path4 = f"{models_dir}/container_lid4.urdf"
        plate = pb_utils.load_model(
            container_path4, pose=pb_utils.Pose(pb_utils.Point(y=-0.4, x=0.4, z=0.01))
        )
        lid1 = pb_utils.load_model(container_path1, fixed_base=False)
        lid2 = pb_utils.load_model(container_path2, fixed_base=False)
        lid3 = pb_utils.load_model(container_path3, fixed_base=False)
        # plate= pb_utils.load_model(container_path3, fixed_base=False)
        # lid4 = pb_utils.load_model(container_path4, fixed_base=False)
        g1= pb_utils.load_model(g1_path, fixed_base=False)
        g2 = pb_utils.load_model(g2_path, fixed_base=False)
        blue_block_path = f"{models_dir}/block_blue.urdf"
        red_block_path = f"{models_dir}/block_red.urdf"
        celery = pb_utils.load_model(blue_block_path, fixed_base=False)
        # cup = pb_utils.load_model(container_path, fixed_base=False)
        radish = pb_utils.load_model(blue_block_path, fixed_base=False)
        egg = pb_utils.load_model(red_block_path, fixed_base=False)
        bacon = pb_utils.load_model(red_block_path, fixed_base=False)

        draw_names = False
        if draw_names:
            pb_utils.add_body_name(g1, "g1", text_size=1)
            pb_utils.add_body_name(g2, "g2", text_size=1)
            pb_utils.add_body_name(lid1, "lid1", text_size=1)
            pb_utils.add_body_name(lid2, "lid2", text_size=1)
            pb_utils.add_body_name(lid3, "lid3", text_size=1)
            pb_utils.add_body_name(north, "mynorth", text_size=1)
            pb_utils.add_body_name(plate, "myplate", text_size=1)
            pb_utils.add_body_name(stove, "mystove", text_size=1)
            pb_utils.add_body_name(sink, "mysink", text_size=1)
            pb_utils.add_body_name(celery, "celery", text_size=1) # blue color
            pb_utils.add_body_name(radish, "radish", text_size=1) #blue color
            pb_utils.add_body_name(egg, "egg", text_size=1)
            pb_utils.add_body_name(bacon, "bacon", text_size=1)
        # cup = pb_utils.load_model('models/dinnerware/cup/cup_small.urdf',
        # pb_utils.Pose(pb_utils.Point(x=+0.5, y=+0.5, z=0.5)), fixed_base=False)
    pb_utils.draw_pose(
        pb_utils.Pose(), parent=robot, parent_link=kuka_primitives.get_tool_link(robot)
    )  # TODO: not working
    # print("KINETMATIC CONF", (robot))
    # dump_body(robot)

    body_names = {
        sink: "mysink",
        stove: "mystove",
        north: "mynorth",
        plate: "myplate",
        celery: "celery",
        radish: "radish",
        bacon: "bacon",
        egg: "egg",
        table: "mytable",
        g1: "g1",
        g2: "g2",
        lid1: "lid1",
        lid2: "lid2",
        lid3: "lid3",
        # lid4: "lid4"
    }
    fixed_bodies = [sink, stove, table, north, plate]
    movable_bodies = [egg, celery, radish, bacon, lid1, lid2, lid3, g1, g2]

    pb_utils.set_pose(
        lid2, pb_utils.Pose(pb_utils.Point(red_box[1],red_box[0], z=pb_utils.stable_z(lid2, sink)))
    )
    pb_utils.set_pose(
        lid3, pb_utils.Pose(pb_utils.Point(green_box[1],green_box[0], z=pb_utils.stable_z(lid3, north)))
    )
    pb_utils.set_pose(
        lid1, pb_utils.Pose(pb_utils.Point(blue_box[1],blue_box[0], z=pb_utils.stable_z(lid1, stove)))
    )
    # pb_utils.set_pose(
    #     plate, pb_utils.Pose(pb_utils.Point(y=-0.3, x=-0.3, z=pb_utils.stable_z(plate, table)))
    # )
    pb_utils.set_pose(
        celery,
        pb_utils.Pose(pb_utils.Point(y=-0.5, x=0.0, z=pb_utils.stable_z(celery, table))), # blue color
    )
    pb_utils.set_pose(
        radish,
        pb_utils.Pose(pb_utils.Point(y=-0.5,x=0.1, z=pb_utils.stable_z(radish, table))), # blue color
    )
    pb_utils.set_pose(
        bacon, pb_utils.Pose(pb_utils.Point( y=-0.7, x=0.1, z=pb_utils.stable_z(bacon, table))) # red color
    )
    pb_utils.set_pose(
        egg,
        pb_utils.Pose(pb_utils.Point( y=-0.7, x=-0, z=pb_utils.stable_z(egg, table))), # red color
    )
    pb_utils.set_pose(
        g1,
        pb_utils.Pose(pb_utils.Point(y=-0.6, x=-0, z=pb_utils.stable_z(g1, table))),  # green color
    )
    pb_utils.set_pose(
        g2,
        pb_utils.Pose(pb_utils.Point(y=-0.6, x=0.1, z=pb_utils.stable_z(g2, table))),  # green color
    )
    new_state = set()
    for m in movable_bodies:
        name = body_names[m]
        new_state.add(f"(On {name} mytable)")

    for t in range(29):
        new_state.add(f"(Next t{t + 1} t{t + 2})")
    new_state.add("(AtTimestep t1)")

    q_home = np.array([0.0, -np.pi / 4, 0.0, -np.pi / 2, 0.0, np.pi / 3, 0.0])
    #q_home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571]  # Franka's default q_home 6 lin

    world = World(
        robot,
        ab,
        body_names,
        movable_bodies,
        fixed_bodies,
        new_state,
        q_home
    )
    write_scene_description(world)
    return world


def init_geometric_state_kuka(use_gui: bool = True) -> Tuple[World, str, Set[str], Set[str], List[Any]]:
    pbt.utils.connect(use_gui=use_gui)
    world = load_world()
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
            pbt.kuka_primitives.BodyConf(
                world.robot, pbt.utils.get_configuration(world.robot)
            ),
        )
    )

    for body_id in world.movable_bodies:
        body_pose = pbt.kuka_primitives.BodyPose(body_id, pbt.utils.get_pose(body_id))
        state.set(world.body_names[body_id], 0, (body_id, body_pose))
        # print("body name and id",world.body_names,body_id)
        stream_state.add(f"AtPose({world.body_names[body_id]}, p0_{body_id})")
        objects.append(coast.Object(f"p0_{body_id}", "pose", body_pose))
        # print("________!!!!!!!!!!!!!!!!!!!!!!!__________")
        # print("body_pose type", type(body_pose))
        # print("body_pose example", body_pose.value)
    return world, state, stream_state, objects

def create_problem(
    goal: str,
    method: str,
    dynamic_timesteps: bool = False,
    timestep_count: Optional[int] = None,
    include_timesteps: bool = True,
) -> None:
    with open("envs/containers/problem_draft.pddl", "r") as draft_file:
        problem_pddl = draft_file.read()
        
    if timestep_count is None:
        if dynamic_timesteps:
            timestep_count = goal.count('(') * 5
        else:
            timestep_count = 30
    timestep_count = max(2, int(timestep_count))
        
    if include_timesteps:
        for i in range(timestep_count - 1):
            problem_pddl += f"\n(Next t{i + 1} t{i + 2})"
    else:
        problem_pddl = problem_pddl.replace("(AtTimestep t1)", "")

    problem_pddl += ")\n(:goal\n(and"
    problem_pddl += goal + "\n"
    problem_pddl += ")))"

    with open("envs/containers/temp_problem.pddl", "w") as f:
        f.write(problem_pddl)

    # add timesteps to domain file as well (COAST mode only)
    if include_timesteps and method.lower() == "coast_z3":
        # coast_z3 domain has a ;TIMESTEPS HERE placeholder that needs filling.
        with open("envs/containers/domain_coast_z3.pddl", "r") as draft_file:
            domain_pddl = draft_file.read()

        timesteps_insertion = ""
        for i in range(timestep_count):
            timesteps_insertion += f"\n        t{i + 1} - timestep"

        domain_pddl = domain_pddl.replace(";TIMESTEPS HERE", timesteps_insertion)

        with open("envs/containers/temp_domain.pddl", "w") as f:
            f.write(domain_pddl)

    return problem_pddl

def main(
    cfg: Any,
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
    use_gui: bool,
    skip_execution: bool,
    no_write: bool,
    continue_from: Optional[str] = None,
    show_thinking: bool = False,
):

    # Remove stale output files so the VLM is prompted fresh on the first experiment.
    # Oracles live in ../oracles/, outside this directory, so they are untouched.
    _output_dir = "envs/containers/prompting/output"
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
        print(f"[containers.run] trial_mode output: {resolved_results_file}")

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
            f"[containers.run] Continuing from {continue_from}, "
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

        goal = config["goals"]
        method = config.get("method", "coast_z3")
        dynamic_timesteps = bool(config.get("dynamic_timesteps", False))
        timestep_count = config.get("timestep_count")

        if config.get("skip", False):
            continue

        use_timesteps = method.lower() in ("coast_fd", "coast_z3")
        problem_pddl = create_problem(
            goal,
            method,
            dynamic_timesteps,
            timestep_count=timestep_count,
            include_timesteps=use_timesteps,
        )
        if method.lower() == "coast_fd":
            domain_pddl = "envs/containers/domain_coast_fd.pddl"
        elif method.lower() == "coast_z3":
            domain_pddl = "envs/containers/temp_domain.pddl"  # built from domain_coast_z3.pddl
        else:
            domain_pddl = "envs/containers/domain_z3.pddl"

        # Print the VLM's chain-of-thought to the terminal (VLMB method only).
        config['show_thinking'] = config.get('show_thinking', show_thinking)

        for e in range(reps):
            # --continue-from: skip experiments already recorded in the CSV.
            if experiment_counter < rows_to_skip:
                experiment_counter += 1
                print(f"[containers.run] Skipping experiment {experiment_counter}/{rows_to_skip} "
                      f"(cfg_idx={cfg_idx}, rep={e})")
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
            ) = init_geometric_state_kuka(use_gui=use_gui)
            print(world)

            # TODO: Miras image saving part
            saver = pbt.utils.WorldSaver()
            pbt.utils.set_renderer(enable=True)
            camera_pose = [1, 1, 1]  # [-0.4, -0.6, 1]
            target_pos = [0, 0, 0]  # perfect angles
            # image = get_image(camera_pos, target_pos, width=1024, height=768, vertical_fov=45.0)
            # Capture image using default camera settings
            image = pbt.utils.get_image(camera_pose, target_pos, width=1024, height=768, vertical_fov=45.0)
            # Save image
            pbt.utils.save_image("envs/containers/prompting/input/default_camera_capture.png",
                                image.rgbPixels)
            
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
                use_cache=False
            )
            print("PLANNING DONE")
            
            # Prepare result data
            res = plan.log.get_timing()
            res['success'] = True
            res['rep_idx'] = e
            res['num_goals'] = config['goals'].count('(')
            if (plan.action_plan is None or plan.objects is None):
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
                    writer_csv = csv.DictWriter(f_object, fieldnames=csv_fieldnames)
                    
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
                    a = 1
                    for action in plan.action_plan:
                        # pbt.utils.wait_for_user("next")
                        # text = pb_utils.add_text(action, (0.6, -0.2, a), text_size=1.7)
                        action.execute(plan.objects)
                        # a -= 0.05
                        if str(action)[0].lower() == 'c':
                            time.sleep(0.3)
                        # pb_utils.remove_debug(text)
            time.sleep(1)
            if pbt.utils.is_connected():
                pbt.utils.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--streams-py",
        nargs="?",
        default="envs/containers/streams.py",
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
        default=10,  # 0.02 1.0 = don't try to resample.
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
        default="envs/containers/config.json",
        help="Configuration file"
    )
    parser.add_argument("--random-seed", type=int, default=3, help="Random seed")
    parser.add_argument("--no-write", action="store_true", help="Skip writing to the results file")
    parser.add_argument("--append", action="store_true", help="Append to the results file")
    parser.add_argument("--use-gui", action="store_true", help="Use GUI")
    parser.add_argument("--skip-execution", action="store_true", help="Skip execution")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Resume from an existing CSV, appending new results and skipping completed experiments")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Print the VLM's chain-of-thought to the terminal (VLMB method only).")
    args = parser.parse_args()
    # print(args)
    try:
        main(**vars(args))
    except Exception as e:
        if pbt.utils.is_connected():
            pbt.utils.disconnect()
        raise e
