from typing import Dict, Optional, Any, List, Tuple

import numpy as np
from ctrlutils import eigen
import spatialdyn as dyn

import coast
from coast import Object
from coast.action import Action
from coast.formula import _and, _forall, _not, _P, _when
from pybullet_tools import kuka_primitives, utils as pb_utils


GEOMETRIC_PREDICATES = [
    _P("AtConf", "?q"),
    _P("Holding", "?b", "?g"),
    _P("AtPose", "?b", "?p"),
]


def compute_opspace_ik(
    ab: dyn.ArticulatedBody,
    pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    ee_offset: np.ndarray = np.array([0.0, 0.0, 0.052]),  # Default for Kuka IIWA 0.106.
) -> List[float]:
    pos = np.array(pose[0])
    quat = eigen.Quaterniond(pose[1])

    # Set nullspace.
    if dyn.opspace.is_singular(ab, dyn.jacobian(ab)):
        q = np.array([np.pi / 2, np.pi / 6, 0, -np.pi / 3, np.pi/2, np.pi / 2])
    else:
        q = np.array(ab.q)
    q[0] = np.arctan2(pos[1], pos[0])

    ab.q = q

    q = dyn.inverse_kinematics(ab, pos, quat, offset=ee_offset)

    return q.tolist()


class Pick(Action):
    action_name="pick"
    parameters = [Object("?b", "object"), Object("?s", "object")]
    inputs = [Object("?q1", "conf"), Object("?p", "pose")]
    outputs = [Object("?g", "grasp"), Object("?q2", "conf"), Object("?traj1",
        "traj"), Object("?traj2", "traj")]
    precondition = _and(_P("AtConf", "?q1"), _P("AtPose", "?b", "?p"))
    effect = _and(
        _not(_P("AtPose", "?b", "?p")),
        _P("Holding", "?b", "?g"),
        _not(_P("AtConf", "?q1")),
        _P("AtConf", "?q2"),
    )
    certified = _and(
        _P("SampleGrasp", "?b", "?g"),
        _P("IK", "?b", "?p", "?g", "?q2", "?traj2"),
        _forall(
            "bb",
            "packable",
            _forall(
                "pp",
                "pose",
                _when(
                    _P("AtPose", "?bb", "?pp"),
                    _P("CFreeTrajPose", "?traj2", "?bb", "?pp"),
                ),
            ),
        ),
        _P("FreeMotion", "?q1", "?q2", "?traj1")
    )

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: List[kuka_primitives.Command] = outputs["traj2"]
        traj2.refine(num_steps=80).execute(time_step=0.0001)
        return True


class Pack(Action):
    action_name = "pack"
    parameters = [Object("?b", "packable"), Object("?box", "box"), Object("?on",
        "object")]
    inputs = [Object("?q1", "conf"), Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q2", "conf"), Object("?traj1",
        "traj"), Object("?traj2", "traj")]
    # Geometric preconditions should only reference geometric state predicates.
    precondition = _and(_P("AtConf", "?q1"), _P("Holding", "?b", "?g"))
    effect = _and(
        _not(_P("Holding", "?b", "?g")),
        _P("AtPose", "?b", "?p"),
        _not(_P("AtConf", "?q1")),
        _P("AtConf", "?q2"),
    )
    certified = _and(
        _P("SamplePose", "?b", "?on", "?p"),
        _P("IK", "?b", "?p", "?g", "?q2", "?traj2"),
        _forall(
            "bb",
            "packable",
            _forall(
                "pp",
                "pose",
                _when(
                    _P("AtPose", "?bb", "?pp"),
                    _P("CFreeTrajPose", "?traj2", "?bb", "?pp"),
                ),
            ),
        ),
        _P("HoldingMotion", "?b", "?g", "?q1", "?q2", "?traj1")
    )

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: List[kuka_primitives.Command] = outputs["traj2"]
        traj2.reverse().refine(num_steps=80).execute(time_step=0.0001)
        return True


class Placelid(Action):
    action_name = "placelid"
    parameters = [Object("?b", "lid"), Object("?box", "box")]
    inputs = [Object("?q1", "conf"), Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q2", "conf"), Object("?traj1",
        "traj"), Object("?traj2", "traj")]
    precondition = _and(_P("AtConf", "?q1"), _P("Holding", "?b", "?g"))
    effect = _and(
        _not(_P("Holding", "?b", "?g")),
        _P("AtPose", "?b", "?p"),
        _not(_P("AtConf", "?q1")),
        _P("AtConf", "?q2"),
    )
    certified = _and(
        _P("SamplePose", "?b", "?box", "?p"),
        _P("IK", "?b", "?p", "?g", "?q2", "?traj2"),
        _forall(
            "bb",
            "packable",
            _forall(
                "pp",
                "pose",
                _when(
                    _P("AtPose", "?bb", "?pp"),
                    _P("CFreeTrajPose", "?traj2", "?bb", "?pp"),
                ),
            ),
        ),
        _P("HoldingMotion", "?b", "?g", "?q1", "?q2", "?traj1")
    )

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: List[kuka_primitives.Command] = outputs["traj2"]
        traj2.reverse().refine(num_steps=80).execute(time_step=0.0001)
        return True


class SamplePose(coast.Stream):
    name = "SamplePose"
    inputs = [Object("?b", "object"), Object("?s", "object")]
    outputs = [Object("?p", "pose")]
    fluents: List[str] = ["AtPose"]
    BOX_Z_OFFSET = -0.20
    BOX_XY_MARGIN = 0.03
    GENERIC_SAMPLE_ATTEMPTS = 200

    def __init__(self, world: Any):
        super().__init__(world, resample_p=0.6)
        self._fixed_bodies = world.fixed_bodies
        self._movable_bodies = world.movable_bodies
        self._robot = world.robot
        self._body_types = world.body_types
        self._box_ids = [bid for bid, btype in self._body_types.items() if btype == "box"]

    def _is_box_surface(self, surface_id: int) -> bool:
        return self._body_types.get(surface_id) == "box"

    def _is_slab(self, body_id: int) -> bool:
        return self._body_types.get(body_id) == "slab"

    def _is_lid(self, body_id: int) -> bool:
        return self._body_types.get(body_id) == "lid"

    def _is_bblock(self, body_id: int) -> bool:
        return self._body_types.get(body_id) == "bblock"

    def _is_sblock(self, body_id: int) -> bool:
        return self._body_types.get(body_id) == "sblock"

    def _apply_atpose_fluents(self, fluents: Dict[str, List[Dict[str, Any]]]) -> None:
        for args in fluents.get("AtPose", []):
            pose = args.get("p")
            if pose is not None:
                pose.assign()

    def _in_collision(
        self,
        body: int,
        fluents: Dict[str, List[Dict[str, Any]]],
        ignore: Optional[set] = None,
    ) -> bool:
        # Evaluate collisions against the projected symbolic state, not just the
        # current simulator state.
        self._apply_atpose_fluents(fluents)
        ignore = ignore or set()
        obstacles = [self._robot] + self._fixed_bodies + self._movable_bodies
        for obstacle in obstacles:
            if obstacle == body or obstacle in ignore:
                continue
            if pb_utils.pairwise_collision(body, obstacle):
                return True
        return False

    def _sample_generic_pose(
        self, body: int, surface: int, fluents: Dict[str, List[Dict[str, Any]]]
    ):
        for _ in range(self.GENERIC_SAMPLE_ATTEMPTS):
            self._apply_atpose_fluents(fluents)
            pose = pb_utils.sample_placement(body, surface)
            if pose is None:
                continue
            pb_utils.set_pose(body, pose)
            if not self._in_collision(body, fluents=fluents, ignore={surface}):
                return kuka_primitives.BodyPose(body, pose)
        return None

    # def _fit_inside_box_xy(self, body: int, surface: int, pose):
    #     """Clamp XY so the full footprint stays inside box bounds with margin."""
    #     pb_utils.set_pose(body, pose)
    #     center, extent = pb_utils.get_center_extent(body)
    #     box_aabb = pb_utils.get_aabb(surface)

    #     half_x = extent[0] / 2.0
    #     half_y = extent[1] / 2.0
    #     min_x = box_aabb.lower[0] + half_x + self.BOX_XY_MARGIN
    #     max_x = box_aabb.upper[0] - half_x - self.BOX_XY_MARGIN
    #     min_y = box_aabb.lower[1] + half_y + self.BOX_XY_MARGIN
    #     max_y = box_aabb.upper[1] - half_y - self.BOX_XY_MARGIN

    #     if min_x > max_x or min_y > max_y:
    #         return None

    #     pos, quat = pose
    #     target_center = np.array(
    #         [
    #             np.clip(pos[0], min_x, max_x),
    #             np.clip(pos[1], min_y, max_y),
    #             pos[2],
    #         ]
    #     )
    #     point = target_center + (pb_utils.get_point(body) - center)
    #     adjusted_pose = ((point[0], point[1], point[2]), quat)
    #     pb_utils.set_pose(body, adjusted_pose)
    #     return adjusted_pose

    def _place_slab_aligned_to_box_center(self, body: int, surface: int):
        """Keep slab XY at box center and align slab orientation with box orientation."""
        if self._is_box_surface(surface):
            box_id = surface
        elif self._box_ids:
            box_id = self._box_ids[0]
        else:
            return None

        support_aabb = pb_utils.get_aabb(surface)
        box_point, box_quat = pb_utils.get_pose(box_id)
        support_top_z = support_aabb.upper[2]
        if self._is_box_surface(surface):
            support_top_z += self.BOX_Z_OFFSET

        # First orient slab to container frame, then compute corrected z.
        seed_pose = ((box_point[0], box_point[1], support_top_z), box_quat)
        pb_utils.set_pose(body, seed_pose)
        center, extent = pb_utils.get_center_extent(body)
        z = support_top_z + extent[2] / 2.0 #+ 1e-5
        point = np.array([box_point[0], box_point[1], z]) + (pb_utils.get_point(body) - center)
        pose = ((point[0], point[1], point[2]), box_quat)
        pb_utils.set_pose(body, pose)
        return pose

    def _place_lid_aligned_to_box_top(self, body: int, surface: int):
        """Place lid centered/aligned with box, resting on top of the container."""
        if self._is_box_surface(surface):
            box_id = surface
        elif self._box_ids:
            box_id = self._box_ids[0]
        else:
            return None

        box_aabb = pb_utils.get_aabb(box_id)
        box_point, box_quat = pb_utils.get_pose(box_id)
        box_top_z = box_aabb.upper[2]

        seed_pose = ((box_point[0], box_point[1], box_top_z), box_quat)
        pb_utils.set_pose(body, seed_pose)
        center, extent = pb_utils.get_center_extent(body)
        z = box_top_z + extent[2] / 2.0 + 3e-3
        point = np.array([box_point[0], box_point[1], z]) + (pb_utils.get_point(body) - center)
        pose = ((point[0], point[1], point[2]), box_quat)
        pb_utils.set_pose(body, pose)
        return pose

    def _sample_block_quadrant_xy(self, body: int, surface: int, stable_z: float):
        # sample from 4 quadrants of xy, make sure collision free
        assert self._is_box_surface(surface) or self._is_slab(surface)
        box_id = surface

        box_point, box_quat = pb_utils.get_pose(box_id)
        aligned_quat = box_quat

        # 4 quadrant centers on the container floor.
        # 0.05 keeps bblocks distributed (2x2) within the box footprint.
        offsets = [(-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05)]

        # if no collision free pose is found, return None
        for idx in np.random.permutation(len(offsets)):
            dx, dy = offsets[idx]
            x = box_point[0] + dx
            y = box_point[1] + dy
            pose = ((x, y, stable_z), aligned_quat)
            pb_utils.set_pose(body, pose)

            # Check collisions against projected world state.
            collides = False
            obstacles = [self._robot] + self._fixed_bodies + self._movable_bodies
            for obstacle in obstacles:
                if obstacle == body or obstacle == surface:
                    continue
                if pb_utils.pairwise_collision(body, obstacle):
                    collides = True
                    break
            if not collides:
                # DO NOT CHANGE STABLE_Z
                return pose
        return None

        


    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body = inputs["b"]
        surface = inputs["s"]
        surface_pose = pb_utils.get_pose(surface)

        # print(f"surface type[{surface}]: {self._body_types.get(surface)}, body type[{body}]: {self._body_types.get(body)}")

        # Slabs stay horizontally centered/aligned with the container even when stacked.
        if self._is_slab(body):
            slab_pose = self._place_slab_aligned_to_box_center(body, surface)
            if slab_pose is None or self._in_collision(
                body, fluents=fluents, ignore={surface}
            ):
                return None
            return {"p": kuka_primitives.BodyPose(body, slab_pose)}

        # Lids are deterministically centered/aligned on top of the container.
        if self._is_lid(body):
            lid_pose = self._place_lid_aligned_to_box_top(body, surface)
            if lid_pose is None or self._in_collision(
                body, fluents=fluents, ignore={surface}
            ):
                return None
            return {"p": kuka_primitives.BodyPose(body, lid_pose)}

        # big-block on slab:
        if self._is_bblock(body) and self._is_slab(surface):
            # determine stable z-value from surface
            stable_z = pb_utils.stable_z(body, surface)
            pose = self._sample_block_quadrant_xy(body, surface, stable_z)
            if pose is None:
                return None
            return {"p": kuka_primitives.BodyPose(body, pose)}

        # block on block:
        if (self._is_bblock(body) and self._is_bblock(surface)) or \
            (self._is_sblock(body) and self._is_sblock(surface)):
            # copy the pose, just change the z
            stable_z = pb_utils.stable_z(body, surface)
            surface_pos, surface_quat = surface_pose
            pose = ((surface_pos[0], surface_pos[1], stable_z), surface_quat)
            pb_utils.set_pose(body, pose)
            body_pose = kuka_primitives.BodyPose(body, pose)
            if self._in_collision(body, fluents=fluents, ignore={surface}):
                return None
            return {"p": body_pose}


        body_pose = self._sample_generic_pose(body, surface, fluents)
        if body_pose is None:
            return None
        
        # on box-floor:
        if self._is_box_surface(surface):
            pos, quat = body_pose.pose
            stable_z = pos[2] + self.BOX_Z_OFFSET

            if self._is_bblock(body) or self._is_sblock(body):
                pose = self._sample_block_quadrant_xy(body, surface, stable_z)

            if pose is None:
                return None
            body_pose = kuka_primitives.BodyPose(body, pose)

        
        if self._in_collision(body, fluents=fluents, ignore={surface}):
            return None
        return {"p": body_pose}


class SampleGrasp(coast.Stream):
    name = "SampleGrasp"
    inputs = [Object("?b", "object")]
    outputs = [Object("?g", "grasp")]
    fluents: List[str] = []

    def __init__(self, world: Any):
        super().__init__(world)
        self._sample_grasp = kuka_primitives.get_grasp_gen(
            robot=world.robot, grasp_name="top"
        )

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body: int = inputs["b"]
        # TODO: save state to avoid grasp generator reset?
        grasp: kuka_primitives.BodyGrasp = next(self._sample_grasp(body=body))
        return None if grasp is None else {"g": grasp[0]}


class FreeMotion(coast.Stream):
    name = "FreeMotion"
    inputs = [Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?traj1", "traj")]
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
            return {"traj1": traj[0]}
        return None


class HoldingMotion(coast.Stream):
    name = "HoldingMotion"
    inputs = [Object("?b", "object"), Object("?g", "grasp"), Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?traj1", "traj")]
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
        # The motion planner carries b via grasp.attachment(), whose assign() runs
        # on every collision check, so planning leaves b wherever the last sampled
        # configuration happened to put it -- measured 0.36m from its placed pose,
        # mid-air.  IK collision-checks against live simulator poses (it declares
        # no AtPose fluents), so that leftover becomes a phantom obstacle for
        # whichever stream runs next.  Put b back where the caller had it.
        saved_pose = pb_utils.get_pose(b)
        try:
            traj = self._sampler(q1, q2, b, g, fluents=atpose_fluents)
        finally:
            pb_utils.set_pose(b, saved_pose)
        if traj is not None:
            return {"traj1": traj[0]}
        return None


class InverseKinematics(coast.Stream):
    name = "IK"
    inputs = [Object("?b", "object"), Object("?p", "pose"), Object("?g", "grasp")]
    outputs = [Object("?q", "conf"), Object("?traj2", "traj")]
    fluents: List[str] = []
    # Minimum height the approach EE pose must be above the grasp EE pose.
    APPROACH_HOVER_Z = 0.15

    def __init__(self, world: Any):
        super().__init__(world)
        self._robot = world.robot
        self._ab: dyn.ArticulatedBody = world.ab
        self._q_home: np.ndarray = world.q_home
        self.world = world
        self._box_ids = [bid for bid, btype in world.body_types.items() if btype == "box"]

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body: int = inputs["b"]
        body_pose: kuka_primitives.BodyPose = inputs["p"]
        grasp: kuka_primitives.BodyGrasp = inputs["g"]
        grasp_pose = pb_utils.end_effector_from_body(body_pose.pose, grasp.grasp_pose)

        # --- Build approach pose: always hover well above the grasp, EE straight down ---
        pos, quat = grasp_pose  # quat already encodes the top-grasp (downward) orientation
        approach_x, approach_y = pos[0], pos[1]
        hover_z = pos[2] + self.APPROACH_HOVER_Z

        # For container placements, center XY over the box and ensure above the rim.
        if self._box_ids:
            box_id = self._box_ids[0]
            box_point, _ = pb_utils.get_pose(box_id)
            box_aabb = pb_utils.get_aabb(box_id)
            in_box_xy = (
                box_aabb.lower[0] <= pos[0] <= box_aabb.upper[0]
                and box_aabb.lower[1] <= pos[1] <= box_aabb.upper[1]
            )
            if in_box_xy:
                approach_x, approach_y = box_point[0], box_point[1]
                hover_z = max(hover_z, box_aabb.upper[2] + 0.02)

        approach_pose = ((approach_x, approach_y, hover_z), quat)

        self._ab.q = self._q_home

        q_approach = compute_opspace_ik(self._ab, approach_pose)

        if np.isnan(q_approach).any():
            return None

        self._ab.q = q_approach
        q_grasp = compute_opspace_ik(self._ab, grasp_pose)

        if np.isnan(q_grasp).any():
            return None

        conf = kuka_primitives.BodyConf(self._robot, q_approach)
        conf.assign()
        obstacles = list(self.world.fixed_bodies) + [
            b for b in self.world.movable_bodies if b != body
        ]
        path = pb_utils.plan_direct_joint_motion(
            self._robot, conf.joints, q_grasp, obstacles=obstacles,
        )

        if path is None:
            return None

        traj = kuka_primitives.Command(
            [
                kuka_primitives.BodyPath(self._robot, path),
                kuka_primitives.Attach(body, self._robot, grasp.link),
                kuka_primitives.BodyPath(self._robot, path[::-1], attachments=[grasp]),
            ]
        )

        return {"q": conf, "traj2": traj}


class TestCFreeTrajPose(coast.Stream):
    name = "CFreeTrajPose"
    inputs = [Object("?t", "traj"), Object("?b", "object"), Object("?p", "pose")]
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
