from typing import Dict, Optional, Any, List, Tuple

import numpy as np
from ctrlutils import eigen
import spatialdyn as dyn

import coast
from coast import Object
from coast.action import Action
from coast.formula import _and, _forall, _not, _P, _when
from pybullet_tools import kuka_primitives, utils as pb_utils


def compute_opspace_ik(
        ab: dyn.ArticulatedBody,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        ee_offset: np.ndarray = np.array([0.0, 0.0, 0.046]),  # Default for Kuka IIWA.
        warm_start: bool = False,
) -> List[float]:
    pos = np.array(pose[0])
    quat = eigen.Quaterniond(pose[1])

    if warm_start:
        # Trust ab.q (the caller's seed) for continuity between nearby
        # waypoints of a Cartesian sweep. Skipping the singularity fallback
        # and the base-joint heuristic below matters here: both are only
        # meant to help an otherwise-unrelated seed (e.g. q_home) converge,
        # but applied every step they'd throw away a perfectly good nearby
        # solution and jump to an unrelated posture mid-path.
        q = dyn.inverse_kinematics(ab, pos, quat, offset=ee_offset)
        return q.tolist()

    # Set nullspace.
    jac = dyn.jacobian(ab)
    if dyn.opspace.is_singular(ab, jac):
        q = np.array([np.pi / 2, np.pi / 6, 0, -np.pi / 3, 0, np.pi / 2, 0])
    else:
        q = np.array(ab.q)

    q[0] = np.arctan2(pos[1], pos[0])
    ab.q = q

    q = dyn.inverse_kinematics(ab, pos, quat, offset=ee_offset)

    return q.tolist()


def plan_cartesian_path(
        ab: dyn.ArticulatedBody,
        robot: int,
        start_q: List[float],
        start_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        end_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        obstacles: List[int],
        resolution: float = 0.02,
) -> Optional[List[List[float]]]:
    """Interpolate a straight line in Cartesian (x, y, z) space between two
    poses that share the same orientation (as the approach/grasp poses do),
    solving IK at each waypoint warm-started from the previous solution.

    Joint-space linear interpolation between two independently-solved IK
    configs can swing the arm through unrelated postures even when the two
    Cartesian endpoints are close together (each is solved from a fresh seed
    with no relation to the other). Stepping through small Cartesian
    increments and warm-starting each IK solve from the previous waypoint
    keeps consecutive joint configs close together, so the end effector --
    and whatever it's carrying -- actually traces the straight line between
    the two poses instead."""
    start_point = np.array(start_pose[0])
    end_point = np.array(end_pose[0])
    quat = start_pose[1]

    distance = float(np.linalg.norm(end_point - start_point))
    num_steps = max(1, int(np.ceil(distance / resolution)))

    joints = pb_utils.get_movable_joints(robot)
    collision_fn = pb_utils.get_collision_fn(robot, joints, obstacles, [], True, set())

    ab.q = np.array(start_q)
    if collision_fn(start_q):
        return None
    path = [start_q]
    for i in range(1, num_steps + 1):
        t = i / num_steps
        point = start_point + t * (end_point - start_point)
        q = compute_opspace_ik(ab, (point, quat), warm_start=True)
        if np.isnan(q).any():
            return None
        if collision_fn(q):
            return None
        ab.q = np.array(q)
        path.append(q)
    return path


GEOMETRIC_PREDICATES = [
    _P("AtConf", "?q"),
    _P("Holding", "?b", "?g"),
    _P("AtPose", "?b", "?p"),
]


class Pick(Action):
    action_name = "pick"
    parameters = [Object("?b", "block"), Object("?loc", "location")]
    inputs = [Object("?q1", "conf"), Object("?p", "pose")]
    outputs = [Object("?g", "grasp"), Object("?q2", "conf"), Object("?t", "traj"), Object("?t2", "traj")]
    precondition = _and(_P("AtConf", "?q1"), _P("AtPose", "?b", "?p"))
    effect = _and(
        _not(_P("AtPose", "?b", "?p")),
        _P("Holding", "?b", "?g"),
        _not(_P("AtConf", "?q1")),
        _P("AtConf", "?q2"),
    )
    certified = _and(
        _P("SampleGrasp", "?b", "?g"),
        _P("IK", "?b", "?p", "?g", "?q2", "?t"),
        _forall(
            "bb",
            "block",
            _forall(
                "pp",
                "pose",
                _when(
                    _P("AtPose", "?bb", "?pp"),
                    _P("CFreeTrajPose", "?t", "?bb", "?pp"),
                ),
            ),
        ),
        _P("FreeMotion", "?q1", "?q2", "?t2")
    )

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj2: List[kuka_primitives.Command] = outputs["t2"]
        traj2.refine(num_steps=60).execute(time_step=0.0001)
        traj: kuka_primitives.Command = outputs["t"]
        traj.refine(num_steps=80).execute(time_step=0.0001)
        return True


class Place(Action):
    action_name = "place"
    parameters = [Object("?b", "block"), Object("?loc", "location")]
    inputs = [Object("?q1", "conf"), Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q2", "conf"), Object("?t", "traj"), Object("?t2", "traj")]
    precondition = _and(_P("AtConf", "?q1"), _P("Holding", "?b", "?g"))
    effect = _and(
        _not(_P("Holding", "?b", "?g")),
        _P("AtPose", "?b", "?p"),
        _not(_P("AtConf", "?q1")),
        _P("AtConf", "?q2"),
    )
    certified = _and(
        _P("SamplePose", "?b", "?loc", "?p"),
        _P("IK", "?b", "?p", "?g", "?q2", "?t"),
        _forall(
            "bb",
            "block",
            _forall(
                "pp",
                "pose",
                _when(
                    _P("AtPose", "?bb", "?pp"),
                    _P("CFreeTrajPose", "?t", "?bb", "?pp"),
                ),
            ),
        ),
        _P("HoldingMotion", "?b", "?g", "?q1", "?q2", "?t2")
    )

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj2: List[kuka_primitives.Command] = outputs["t2"]
        traj2.refine(num_steps=60).execute(time_step=0.0001)
        traj: kuka_primitives.Command = outputs["t"]
        b = traj.body_paths
        new_t = kuka_primitives.Command(b[:])
        new_t.reverse().refine(num_steps=80).execute(time_step=0.0001)
        return True


class SamplePose(coast.Stream):
    name = "SamplePose"
    inputs = [Object("?b", "block"), Object("?loc", "location")]
    outputs = [Object("?p", "pose")]
    fluents: List[str] = []

    def __init__(self, world: Any):
        super().__init__(world)
        self._sample_pose = kuka_primitives.get_stable_gen(fixed=world.fixed_bodies)

    def sample(
            self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        pos, quat = inputs["loc"]
        pose = kuka_primitives.BodyPose(
            body=inputs["b"], pose=pb_utils.Pose(pb_utils.Point(*pos))
        )
        return {"p": pose}


class SampleGrasp(coast.Stream):
    name = "SampleGrasp"
    inputs = [Object("?b", "block")]
    outputs = [Object("?g", "grasp")]
    fluents: List[str] = []

    def __init__(self, world: Any):
        super().__init__(world)
        self._sample_grasp = kuka_primitives.get_grasp_gen(
            robot=world.robot, grasp_name="side"
        )

    def sample(
            self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body: int = inputs["b"]
        # TODO: save state to avoid grasp generator reset?
        grasp: kuka_primitives.BodyGrasp = next(self._sample_grasp(body=body))[0]
        return None if grasp is None else {"g": grasp}


class InverseKinematics(coast.Stream):
    name = "IK"
    inputs = [Object("?b", "block"), Object("?p", "pose"), Object("?g", "grasp")]
    outputs = [Object("?q", "conf"), Object("?t", "traj")]
    fluents: List[str] = []

    def __init__(self, world: Any):
        super().__init__(world)
        self._robot = world.robot
        self._ab: dyn.ArticulatedBody = world.ab
        self._q_home: np.ndarray = world.q_home
        self._obstacles = world.fixed_bodies  # shelf + table, so the dive/retreat can't clip either
        self._retract_y = world.shelf_retract_y

    def sample(
            self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body_pose: kuka_primitives.BodyPose = inputs["p"]
        grasp: kuka_primitives.BodyGrasp = inputs["g"]

        grasp_pose = pb_utils.end_effector_from_body(body_pose.pose, grasp.grasp_pose)
        approach_pose = pb_utils.approach_from_grasp(grasp.approach_pose, grasp_pose)
        if self._retract_y is not None:
            # Force the approach/retreat standoff to a fixed y clear of the
            # shelf opening (independent of grid size, see add_grid_shelf in
            # run.py), so the arm always fully exits the shelf on every
            # pick/place instead of stopping just short of the lid/walls.
            # This keeps FreeMotion/HoldingMotion transit endpoints in open
            # space, which the RRT planner can connect far more reliably than
            # endpoints still tucked under the lid.
            point, quat = approach_pose
            approach_pose = ((point[0], self._retract_y, point[2]), quat)

        self._ab.q = self._q_home
        q_approach = compute_opspace_ik(self._ab, approach_pose)
        if np.isnan(q_approach).any():
            return None

        conf = kuka_primitives.BodyConf(self._robot, q_approach)
        conf.assign()

        # VIZ-COAST EDITED
        path = plan_cartesian_path(
            self._ab, self._robot, q_approach, approach_pose, grasp_pose, self._obstacles
        )

        if path is None:
            return None

        traj = kuka_primitives.Command(
            [
                kuka_primitives.BodyPath(self._robot, path),
                kuka_primitives.Attach(object, self._robot, grasp.link),
                kuka_primitives.BodyPath(self._robot, path[::-1], attachments=[grasp]),
            ]
        )

        return {"q": conf, "t": traj}


class FreeMotion(coast.Stream):
    name = "FreeMotion"
    inputs = [Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?t2", "traj")]
    fluents: List[str] = ["AtPose"]

    def __init__(self, world: Any):
        super().__init__(world)
        self._robot = world.robot
        self._ab: dyn.ArticulatedBody = world.ab
        self._q_home: np.ndarray = world.q_home
        self._sampler = kuka_primitives.get_free_motion_gen(self._robot, world.fixed_bodies)

    def sample(
            self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        q1 = inputs["q1"]
        q2 = inputs["q2"]
        atpose_fluents = [("atpose", args["b"], args["p"]) for args in fluents["AtPose"]]
        traj = self._sampler(q1, q2, fluents=atpose_fluents)
        if traj is not None:
            return {"t2": traj[0]}
        return None


class HoldingMotion(coast.Stream):
    name = "HoldingMotion"
    inputs = [Object("?b", "object"), Object("?g", "grasp"), Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?t2", "traj")]
    fluents: List[str] = ["AtPose"]

    def __init__(self, world: Any):
        super().__init__(world)
        self._robot = world.robot
        self._ab: dyn.ArticulatedBody = world.ab
        self._q_home: np.ndarray = world.q_home
        self._sampler = kuka_primitives.get_holding_motion_gen(self._robot, world.fixed_bodies)

    def sample(
            self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        q1 = inputs["q1"]
        q2 = inputs["q2"]
        b = inputs["b"]
        g = inputs["g"]
        atpose_fluents = [("atpose", args["b"], args["p"]) for args in fluents["AtPose"]]
        traj = self._sampler(q1, q2, b, g, fluents=atpose_fluents)
        if traj is not None:
            return {"t2": traj[0]}
        return None


class TestCFreeTrajPose(coast.Stream):
    name = "CFreeTrajPose"
    inputs = [Object("?t", "traj"), Object("?b", "block"), Object("?p", "pose")]
    outputs: List[str] = []
    fluents: List[str] = []

    def __init__(self, world: Any):
        super().__init__(world)
        self._test = kuka_primitives.get_movable_collision_test()

    def sample(
            self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        command: kuka_primitives.Command = inputs["t"]
        body: int = inputs["b"]
        pose: kuka_primitives.BodyPose = inputs["p"]
        assert pose.body == body

        does_collide = self._test(command=command, body=body, pose=pose)

        return None if does_collide else {}
