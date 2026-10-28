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
    parameters = [Object("?b", "object"), Object("?s", "phys")]
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
            "block",
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


class Place(Action):
    action_name = "place"
    parameters = [Object("?b", "object"), Object("?s", "phys")]
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
        _P("SamplePose", "?b", "?s", "?p"),
        _P("IK", "?b", "?p", "?g", "?q2", "?traj2"),
        _forall(
            "bb",
            "block",
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


class Placetop(Action):
    action_name = "placetop"
    parameters = [Object("?b", "object2"), Object("?s", "phys")]
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
        _P("SamplePose", "?b", "?s", "?p"),
        _P("IK", "?b", "?p", "?g", "?q2", "?traj2"),
        _forall(
            "bb",
            "block",
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


class Unplacetop(Action):
    action_name = "unplacetop"
    parameters = [Object("?b", "object2"), Object("?s", "phys")]
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
            "block",
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

    #IDK WHY THIS WASNT HERE BEFORE
    #it was causing the "teleporting"
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: List[kuka_primitives.Command] = outputs["traj2"]
        traj2.reverse().refine(num_steps=80).execute(time_step=0.0001)
        return True


class SamplePose(coast.Stream):
    name = "SamplePose"
    inputs = [Object("?b", "object"), Object("?s", "phys")]
    outputs = [Object("?p", "pose")]
    fluents: List[str] = []

    # Surface names that are containers (blocks should sit inside, not on top).
    CONTAINER_SURFACES = {"mysink", "mystove", "mynorth"}
    # Lids are best handled by deterministic centered placements on these surfaces.
    CENTER_LID_SURFACES = CONTAINER_SURFACES | {"myplate"}
    GENERIC_SAMPLE_ATTEMPTS = 200
    # Keep sampled placements this far clear of the container's inner walls.
    WALL_MARGIN = 0.005
    # Rest placements this far above the container floor so the resting contact
    # does not register as a penetration.
    FLOOR_CLEARANCE = 1e-3

    def __init__(self, world: Any):
        super().__init__(world, resample_p=0.6)
        self._fixed_bodies = world.fixed_bodies
        self._movable_bodies = world.movable_bodies
        self._robot = world.robot
        self._body_names = world.body_names
        # Container interiors are static geometry; compute them once per body.
        self._interior_cache: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
        # print("BODY NAMES", self._body_names)

    def _is_lid(self, body_id: int) -> bool:
        return self._body_names.get(body_id, "").startswith("lid")

    def _is_container(self, surface_id: int) -> bool:
        return self._body_names.get(surface_id, "") in self.CONTAINER_SURFACES

    def _is_center_lid_surface(self, surface_id: int) -> bool:
        return self._body_names.get(surface_id, "") in self.CENTER_LID_SURFACES

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

    def _container_interior(self, surface: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (lower_xy, upper_xy, floor_z) of the space inside a container.

        A container is a single body whose base_link is the floor slab and
        whose remaining links are the four walls.  sample_placement() draws
        from the body's *outer* AABB, which spans the walls too, so a
        placement taken from there can land embedded in one -- invisible to
        the collision check below, which has to ignore the support surface,
        but fatal later when IK collision-checks against the same walls.
        Deriving the interior from the per-link AABBs avoids that entirely.
        """
        try:
            return self._interior_cache[surface]
        except KeyError:
            pass

        floor = pb_utils.get_aabb(surface, link=pb_utils.BASE_LINK)
        lower = np.array(floor[0][:2], dtype=float)
        upper = np.array(floor[1][:2], dtype=float)
        center_xy = (lower + upper) / 2.0

        for link in pb_utils.get_all_links(surface):
            if link == pb_utils.BASE_LINK:
                continue
            wall = pb_utils.get_aabb(surface, link=link)
            # A wall is a slab; its thin horizontal axis is the one it faces along.
            axis = int(np.argmin(pb_utils.get_aabb_extent(wall)[:2]))
            if (wall[0][axis] + wall[1][axis]) / 2.0 < center_xy[axis]:
                lower[axis] = max(lower[axis], wall[1][axis])
            else:
                upper[axis] = min(upper[axis], wall[0][axis])

        interior = (lower, upper, float(floor[1][2]))
        self._interior_cache[surface] = interior
        return interior

    def _sample_container_pose(
        self, body: int, surface: int, fluents: Dict[str, List[Dict[str, Any]]]
    ):
        """Sample a placement resting on a container's floor, inside its walls.

        XY is drawn with a triangular distribution peaked at the container
        centre rather than uniformly: the wrist has to descend into a narrow
        opening to reach the object, so placements hugging a wall are usually
        unreachable even though they are collision-free.  Biasing keeps the
        whole interior reachable in principle while spending most draws where
        IK actually succeeds.
        """
        lower_xy, upper_xy, floor_z = self._container_interior(surface)

        for _ in range(self.GENERIC_SAMPLE_ATTEMPTS):
            rotation = pb_utils.Euler(yaw=np.random.uniform(-np.pi, np.pi))
            pb_utils.set_pose(body, pb_utils.multiply(
                pb_utils.Pose(euler=rotation), pb_utils.unit_pose()
            ))
            center, extent = pb_utils.get_center_extent(body)

            half = extent[:2] / 2.0 + self.WALL_MARGIN
            lo = lower_xy + half
            hi = upper_xy - half
            if np.any(lo >= hi):
                continue  # Body does not fit inside the walls at this yaw.

            xy = np.random.triangular(lo, (lo + hi) / 2.0, hi)
            z = floor_z + extent[2] / 2.0 + self.FLOOR_CLEARANCE
            point = np.array([xy[0], xy[1], z]) + (pb_utils.get_point(body) - center)
            pose = pb_utils.multiply(
                pb_utils.Pose(point, rotation), pb_utils.unit_pose()
            )
            pb_utils.set_pose(body, pose)

            # The container stays in the collision check here: unlike a flat
            # support surface, its walls are exactly what we need to avoid.
            if not self._in_collision(body, fluents=fluents):
                return kuka_primitives.BodyPose(body, pose)

        print("container sample failed")
        return None

    def _sample_generic_pose(self, body: int, surface: int, fluents: Dict[str, List[Dict[str, Any]]]):
        """Bounded-time stable placement sampler with explicit collision checks."""
        for _ in range(self.GENERIC_SAMPLE_ATTEMPTS):
            pose = pb_utils.sample_placement(body, surface)
            if pose is None:
                continue
            pb_utils.set_pose(body, pose)
            if not self._in_collision(body, fluents=fluents, ignore={surface}):
                return kuka_primitives.BodyPose(body, pose)
        print("generic sample failed")
        return None

    # OVERRIDE STANDARD SAMPLER TO SAMPLE IN CONTAINERS
    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body = inputs["b"]
        surface = inputs["s"]

        if self._is_lid(body) and self._is_center_lid_surface(surface):
            # Center lids on designated surfaces for robust deterministic sampling.
            pose = self._center_on_surface(body, surface)
            if pose is None or self._in_collision(body, fluents=fluents, ignore={surface}):
                return None
            body_pose = kuka_primitives.BodyPose(body, pose)
            return {"p": body_pose}

        # Do not do generic/random sampling for lids on disallowed surfaces.
        if self._is_lid(body):
            return None

        if self._is_container(surface):
            body_pose = self._sample_container_pose(body, surface, fluents)
            return None if body_pose is None else {"p": body_pose}

        # Use bounded generic sampling for all remaining cases.
        body_pose = self._sample_generic_pose(body, surface, fluents)
        if body_pose is None:
            return None

        if self._in_collision(body, fluents=fluents, ignore={surface}):
            return None

        return {"p": body_pose}

    # HELPER FUNCTION TO CENTER OBJECT ON SURFACE
    @staticmethod
    def _center_on_surface(body: int, surface: int):
        """Place body centered on surface with no rotation."""
        bottom_aabb = pb_utils.get_aabb(surface)
        rotation = pb_utils.Euler(yaw=0.0)
        pb_utils.set_pose(body, pb_utils.multiply(
            pb_utils.Pose(euler=rotation), pb_utils.unit_pose()
        ))
        center, extent = pb_utils.get_center_extent(body)
        x = (bottom_aabb[0][0] + bottom_aabb[1][0]) / 2
        y = (bottom_aabb[0][1] + bottom_aabb[1][1]) / 2
        z = (bottom_aabb[1] + extent / 2.)[2] + 1e-3
        point = np.array([x, y, z]) + (pb_utils.get_point(body) - center)
        pose = pb_utils.multiply(
            pb_utils.Pose(point, rotation), pb_utils.unit_pose()
        )
        pb_utils.set_pose(body, pose)
        return pose


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
        traj = self._sampler(q1, q2, b, g, fluents=atpose_fluents)
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
    # Container surface names (match SamplePose.CONTAINER_SURFACES).
    CONTAINER_SURFACES = {"mysink", "mystove", "mynorth"}

    # NOTE: this stream is a deterministic function of (body, pose, grasp), so
    # resampling the same instance can never change its outcome.  Randomising
    # the IK seed to explore the 7-DoF null space was tried and measured at
    # +1 success in 320 paired trials for ~2x the runtime: when a container
    # grasp fails it is iiwa_link_6/link_7 hitting the wall, and those links
    # are pinned by the EE pose, which no null-space motion can move.  The
    # lever for these failures is the placement distribution (see SamplePose),
    # not the IK solver.

    def __init__(self, world: Any):
        super().__init__(world)
        self._robot = world.robot
        self._ab: dyn.ArticulatedBody = world.ab
        self._q_home: np.ndarray = world.q_home
        self.world = world
        self._body_names = world.body_names
        self._container_ids = [
            bid for bid, bname in world.body_names.items()
            if bname in self.CONTAINER_SURFACES
        ]
        self._block_ids = ["g1", "g2", "bacon", "egg", "radish", "celery"]

    def _is_lid(self, body_id: int) -> bool:
        return self._body_names.get(body_id, "").startswith("lid")

    def _is_block(self, body_id: int) -> bool:
        return self._body_names.get(body_id, "") in self._block_ids

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body: int = inputs["b"]
        body_pose: kuka_primitives.BodyPose = inputs["p"]
        grasp: kuka_primitives.BodyGrasp = inputs["g"]
        grasp_pose = pb_utils.end_effector_from_body(body_pose.pose, grasp.grasp_pose)

        if self._is_block(body):
            # print("sampling IK block")
            # Lids use the default small approach offset (no high hover).

            # --- Build approach pose: hover well above the grasp, EE straight down ---
            pos, quat = grasp_pose  # quat already encodes the top-grasp (downward) orientation
            approach_x, approach_y = pos[0], pos[1]
            hover_z = pos[2] + self.APPROACH_HOVER_Z

            # For container placements, center XY over the container and ensure above the rim.
            # for cid in self._container_ids:
            #     c_point, _ = pb_utils.get_pose(cid)
            #     c_aabb = pb_utils.get_aabb(cid)
            #     in_container_xy = (
            #         c_aabb.lower[0] <= pos[0] <= c_aabb.upper[0]
            #         and c_aabb.lower[1] <= pos[1] <= c_aabb.upper[1]
            #     )
            #     if in_container_xy:
            #         approach_x, approach_y = c_point[0], c_point[1]
            #         hover_z = max(hover_z, c_aabb.upper[2] + 0.02)
            #         break

            approach_pose = ((approach_x, approach_y, hover_z), quat)

        else:
            approach_pose = pb_utils.approach_from_grasp(grasp.approach_pose, grasp_pose)


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
        # Collision-check the approach segment against scene geometry.
        # Exclude the grasped/target body so contact at grasp is still feasible.
        obstacles = list(self.world.fixed_bodies) + [
            b for b in self.world.movable_bodies if b != body
        ]
        path = pb_utils.plan_direct_joint_motion(
            self._robot, conf.joints, q_grasp, obstacles=obstacles
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
