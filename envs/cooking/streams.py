import os
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
    _P("AtConf", "?a", "?q"),
    _P("Holding", "?a", "?b", "?g"),
    _P("AtPose", "?b", "?p"),
]

EE_OFFSET = np.array([0.0, 0.0, 0.052])  # Default for Kuka IIWA 0.106.
# spatialdyn's own kTolerancePos is 1e-3; allow a little slack before calling
# an IK solution a miss.
IK_POS_TOL = 5e-3


def compute_opspace_ik(
    ab: dyn.ArticulatedBody,
    pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    ee_offset: np.ndarray = EE_OFFSET,
) -> List[float]:
    """Solve IK for a pose expressed in `ab`'s OWN base frame.

    dyn.urdf.load_model builds the articulated body with its base at the
    identity, so callers with a robot mounted away from the world origin must
    push the pose through the base transform first -- see solve_arm_ik().
    """
    pos = np.array(pose[0])
    quat = eigen.Quaterniond(pose[1])

    # Set nullspace.
    if dyn.opspace.is_singular(ab, dyn.jacobian(ab)):
        # 7 entries: the iiwa has 7 DoF.  (packing/containers carry a 6-entry
        # version of this vector, which would raise on assignment; it has never
        # fired there because they only ever seed from q_home.)
        q = np.array([0.0, np.pi / 6, 0.0, -np.pi / 3, 0.0, np.pi / 2, 0.0])
    else:
        q = np.array(ab.q)
    # Point joint 1 (base yaw) at the target.  Only meaningful in base frame.
    q[0] = np.arctan2(pos[1], pos[0])

    ab.q = q

    q = dyn.inverse_kinematics(ab, pos, quat, offset=ee_offset)

    return q.tolist()


def solve_arm_ik(
    arm: Any,
    world_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    q_seed: np.ndarray,
    ee_offset: np.ndarray = EE_OFFSET,
) -> Optional[List[float]]:
    """World-frame IK for one arm, with a hard reachability check.

    spatial_dyn::InverseKinematics is a fixed-step PD simulation with no
    success flag: handed an unreachable target it simply returns wherever the
    integrator drifted to.  np.isnan() catches only the subset that blows up at
    a singularity.  Without the forward-kinematics residual gate below, "the
    left arm cannot reach the right pot" would silently produce a plausible
    joint vector and a collision-free trajectory whose end effector is nowhere
    near the object -- i.e. the domain's central geometric constraint would not
    be enforced at all.
    """
    pose_b = pb_utils.multiply(arm.inv_base_pose, world_pose)

    arm.ab.q = q_seed
    q = compute_opspace_ik(arm.ab, pose_b, ee_offset)
    if np.isnan(q).any():
        return None

    x = np.asarray(dyn.position(arm.ab, offset=ee_offset, q=np.asarray(q)))
    if np.linalg.norm(x - np.asarray(pose_b[0])) > IK_POS_TOL:
        return None
    return q


# kuka_primitives' top-grasp generator routes through approximate_as_prism,
# which asserts the body is single-link.  The lid carries its knob as a second
# link, so build the same grasp directly from the whole-body extent.  Mirrors
# get_top_grasps(under=True, tool_pose=Pose(), grasp_length=-0.001).
_GRASP_LENGTH = -0.001


def _top_grasp_pose(body: int):
    # Measure at the identity pose: get_top_grasps works from the body-frame
    # extent, whereas get_center_extent reports the CURRENT world AABB.  Using
    # the world centre offsets the grasp by wherever the body happens to be.
    saved = pb_utils.get_pose(body)
    try:
        pb_utils.set_pose(body, pb_utils.unit_pose())
        center, extent = pb_utils.get_center_extent(body)
    finally:
        pb_utils.set_pose(body, saved)
    reflect_z = pb_utils.Pose(euler=[0, np.pi, 0])
    translate_z = pb_utils.Pose(point=[0, 0, extent[2] / 2.0 - _GRASP_LENGTH])
    translate_center = pb_utils.Pose(
        point=pb_utils.point_from_pose(pb_utils.unit_pose()) - center
    )
    rotate_z = pb_utils.Pose(euler=[0, 0, 0])
    return pb_utils.multiply(
        pb_utils.Pose(), translate_z, rotate_z, reflect_z,
        translate_center, pb_utils.unit_pose(),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _cfree_traj(traj: str) -> Tuple[Any, Any]:
    """Collision-test `traj` against every placed ingredient and lid.

    Two sibling foralls because coast's _forall matches Object.type by exact
    string equality -- there is no subtype walk, so one forall cannot cover
    both leaf types.  Omitting the lids (as packing does) would be fatal here:
    the lid is this domain's primary dynamic obstacle.
    """
    return (
        _forall(
            "bb",
            "ingredient",
            _forall(
                "pp",
                "pose",
                _when(_P("AtPose", "?bb", "?pp"), _P("CFreeTrajPose", traj, "?bb", "?pp")),
            ),
        ),
        _forall(
            "ll",
            "lid",
            _forall(
                "pp2",
                "pose",
                _when(_P("AtPose", "?ll", "?pp2"), _P("CFreeTrajPose", traj, "?ll", "?pp2")),
            ),
        ),
    )


# How far off a pot's rim a lid may sit and still count as closing it.  It has
# to absorb the drift between the planned placement and where the trajectory
# actually leaves the body -- measured at ~8 mm -- while staying far below the
# ~0.15 m a lid is lifted to before it is carried anywhere.
_RIM_TOLERANCE = 0.02


def _pot_interior_bounds(pot: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """(lower_xy, upper_xy, floor_z) of the space inside a pot's walls.

    A pot is a single body whose base_link is the floor slab and whose
    remaining links are the four walls, so the body's outer AABB describes the
    *column over* the pot, not the space inside it.  The interior has to come
    from the per-link AABBs.
    """
    floor = pb_utils.get_aabb(pot, link=pb_utils.BASE_LINK)
    lower = np.array(floor[0][:2], dtype=float)
    upper = np.array(floor[1][:2], dtype=float)
    center_xy = (lower + upper) / 2.0

    for link in pb_utils.get_all_links(pot):
        if link == pb_utils.BASE_LINK:
            continue
        wall = pb_utils.get_aabb(pot, link=link)
        # A wall is a slab; its thin horizontal axis is the one it faces along.
        axis = int(np.argmin(pb_utils.get_aabb_extent(wall)[:2]))
        if (wall[0][axis] + wall[1][axis]) / 2.0 < center_xy[axis]:
            lower[axis] = max(lower[axis], wall[1][axis])
        else:
            upper[axis] = min(upper[axis], wall[0][axis])

    return lower, upper, float(floor[1][2])


def _pot_containing(world: Any, body: int) -> Optional[int]:
    """The pot `body` is down inside, or None.

    Actions need this because execute() is handed only the geometric inputs and
    outputs -- the ?on parameter naming the target surface never reaches it --
    so "was this put in the fryer?" has to be answered from where the body
    actually landed.

    Answering it with the pot's *footprint* is what a naive version does, and
    it is wrong in a way this domain walks straight into: a lid covering a pot
    occupies the same XY column as the pot, so footprint containment is equally
    true of an ingredient resting on that lid, a clear 0.03 m above the rim.
    Cooking must not follow from being above a closed pot -- an ingredient is
    fried only where the arm could actually have put it, inside the walls and
    down below the rim.

    Sunk below the rim rather than in contact with the floor: the floor is
    where SamplePose puts things, but the trajectory leaves the body a few
    millimetres off its planned pose, and a contact test tight enough to mean
    anything would be within that error.  The rim is 0.09 m above a resting
    ingredient and 0.04 m below one on the lid, so it separates the two cases
    with room to spare.
    """
    aabb = pb_utils.get_aabb(body)
    centre_xy = ((aabb[0][0] + aabb[1][0]) / 2.0, (aabb[0][1] + aabb[1][1]) / 2.0)
    for pot, body_type in world.body_types.items():
        if body_type != "pot":
            continue
        lower_xy, upper_xy, _ = _pot_interior_bounds(pot)
        if not (
            lower_xy[0] <= centre_xy[0] <= upper_xy[0]
            and lower_xy[1] <= centre_xy[1] <= upper_xy[1]
        ):
            continue
        if aabb[0][2] < pb_utils.get_aabb(pot).upper[2]:
            return pot
    return None


def _lid_covering(world: Any, lid: int) -> Optional[int]:
    """The pot `lid` is closing, or None.

    A lid closes a pot by resting on its rim, so unlike an ingredient it is
    never in the interior and needs its own test.  Lid and pot share one
    0.27 x 0.27 footprint, so a lid over a pot is centred on it; what
    distinguishes closing the pot from merely hanging above it is that the
    lid's slab is down on the rim.
    """
    slab = pb_utils.get_aabb(lid, link=pb_utils.BASE_LINK)
    centre_xy = ((slab[0][0] + slab[1][0]) / 2.0, (slab[0][1] + slab[1][1]) / 2.0)
    for pot, body_type in world.body_types.items():
        if body_type != "pot":
            continue
        aabb = pb_utils.get_aabb(pot)
        if not (
            aabb.lower[0] <= centre_xy[0] <= aabb.upper[0]
            and aabb.lower[1] <= centre_xy[1] <= aabb.upper[1]
        ):
            continue
        if abs(slab[0][2] - aabb.upper[2]) <= _RIM_TOLERANCE:
            return pot
    return None


def _recolor_cooked(world: Any, body: int, status: str) -> None:
    """Darken `body` if `status` is the cooked state it is meant to reach.

    Colour encodes the GOAL status (see run.py STATUS_COLORS), so only the
    cooking an ingredient was actually assigned changes its shade: frying
    something destined for the steamer leaves it light blue, which reads as
    "still not done" rather than silently claiming success.
    """
    goal = world.goal_assignment.get(world.body_names.get(body))
    if goal is None or goal[1] != status:
        return
    pb_utils.set_color(body, world.cooked_colors[status])


class Pick(Action):
    action_name = "pick"
    parameters = [Object("?a", "arm"), Object("?b", "object"), Object("?s", "object")]
    inputs = [Object("?q1", "conf"), Object("?p", "pose")]
    outputs = [Object("?g", "grasp"), Object("?q2", "conf"), Object("?traj1",
        "traj"), Object("?traj2", "traj")]
    precondition = _and(_P("AtConf", "?a", "?q1"), _P("AtPose", "?b", "?p"))
    effect = _and(
        _not(_P("AtPose", "?b", "?p")),
        _P("Holding", "?a", "?b", "?g"),
        _not(_P("AtConf", "?a", "?q1")),
        _P("AtConf", "?a", "?q2"),
    )
    certified = _and(
        _P("SampleGrasp", "?a", "?b", "?g"),
        _P("IK", "?a", "?b", "?p", "?g", "?q2", "?traj2"),
        *_cfree_traj("?traj2"),
        _P("FreeMotion", "?a", "?q1", "?q2", "?traj1"),
    )

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: kuka_primitives.Command = outputs["traj2"]
        traj2.refine(num_steps=80).execute(time_step=0.0001)
        return True


class Place(Action):
    action_name = "place"
    parameters = [Object("?a", "arm"), Object("?b", "ingredient"), Object("?on", "object")]
    inputs = [Object("?q1", "conf"), Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q2", "conf"), Object("?traj1",
        "traj"), Object("?traj2", "traj")]
    precondition = _and(_P("AtConf", "?a", "?q1"), _P("Holding", "?a", "?b", "?g"))
    effect = _and(
        _not(_P("Holding", "?a", "?b", "?g")),
        _P("AtPose", "?b", "?p"),
        _not(_P("AtConf", "?a", "?q1")),
        _P("AtConf", "?a", "?q2"),
    )
    certified = _and(
        _P("SamplePose", "?b", "?on", "?p"),
        _P("IK", "?a", "?b", "?p", "?g", "?q2", "?traj2"),
        *_cfree_traj("?traj2"),
        _P("HoldingMotion", "?a", "?b", "?g", "?q1", "?q2", "?traj1"),
    )

    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: kuka_primitives.Command = outputs["traj2"]
        traj2.reverse().refine(num_steps=80).execute(time_step=0.0001)
        # Mirrors the domain's `(when (isfryer ?on) (fried ?obj))`.  ?on is the
        # fryer only when the ingredient ended up in the pot itself; landing on
        # the lid that covers it is `(on ?obj lidleft)`, which fries nothing.
        body_pose: kuka_primitives.BodyPose = outputs["p"]
        if _pot_containing(self.world, body_pose.body) == self.world.fryer:
            _recolor_cooked(self.world, body_pose.body, "fried")
        return True


class Placelid(Action):
    action_name = "placelid"
    parameters = [Object("?a", "arm"), Object("?b", "lid"), Object("?on", "object")]
    inputs = [Object("?q1", "conf"), Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q2", "conf"), Object("?traj1",
        "traj"), Object("?traj2", "traj")]
    precondition = _and(_P("AtConf", "?a", "?q1"), _P("Holding", "?a", "?b", "?g"))
    effect = _and(
        _not(_P("Holding", "?a", "?b", "?g")),
        _P("AtPose", "?b", "?p"),
        _not(_P("AtConf", "?a", "?q1")),
        _P("AtConf", "?a", "?q2"),
    )
    certified = _and(
        _P("SamplePose", "?b", "?on", "?p"),
        _P("IK", "?a", "?b", "?p", "?g", "?q2", "?traj2"),
        *_cfree_traj("?traj2"),
        _P("HoldingMotion", "?a", "?b", "?g", "?q1", "?q2", "?traj1"),
    )

    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        traj1: kuka_primitives.Command = outputs["traj1"]
        traj1.refine(num_steps=60).execute(time_step=0.0001)
        traj2: kuka_primitives.Command = outputs["traj2"]
        traj2.reverse().refine(num_steps=80).execute(time_step=0.0001)
        # Mirrors the domain's forall-when on placelid: covering the steamer
        # steams everything already inside it.  Membership is read from the
        # live simulator, which at execution time is the state the plan has
        # actually reached.  Both halves are containment tests rather than
        # footprint ones -- `(on ?i ?on)` holds for what is in the pot, not for
        # an ingredient the other arm happens to be holding over it, nor for
        # one parked on the lid going on.
        body_pose: kuka_primitives.BodyPose = outputs["p"]
        if _lid_covering(self.world, body_pose.body) == self.world.steamer:
            for body, body_type in self.world.body_types.items():
                if body_type != "ingredient":
                    continue
                if _pot_containing(self.world, body) == self.world.steamer:
                    _recolor_cooked(self.world, body, "steamed")
        return True


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

class _ArmMixin:
    """Mixin for streams that act with a specific arm.

    The `?a` input arrives as the arm's PyBullet body id, because run.py
    registers coast.Object(arm.name, "arm", arm.robot) and retrace() resolves
    the PDDL constant through that map.

    Deliberately NOT a coast.Stream subclass: coast.main.load_streams
    instantiates every Stream subclass it finds in this module's namespace, and
    an abstract intermediate base would blow up there.
    """

    def __init__(self, world: Any, **kwargs: Any):
        super().__init__(world, **kwargs)
        self.world = world
        self._arms = world.arms_by_id
        # Used only to clear an idle arm off the shared plate.  A straight
        # joint-space line from a hover pose back to park sweeps through the
        # table, so the retreat needs the same planner a normal transit uses.
        self._retreat_free = {
            a.robot: kuka_primitives.get_free_motion_gen(a.robot, a.fixed_bodies)
            for a in world.arms.values()
        }
        self._retreat_hold = {
            a.robot: kuka_primitives.get_holding_motion_gen(a.robot, a.fixed_bodies)
            for a in world.arms.values()
        }

    def arm(self, inputs: Dict[str, Any]):
        return self._arms[inputs["a"]]

    # Actions on the shared plate are the one place the two arms genuinely
    # contend for space: an arm that has just picked from platemid finishes at
    # its approach configuration, hovering directly over it.  park_others()
    # below plans as though the idle arm were tucked away, which keeps
    # collision checking deterministic but is a fiction unless the idle arm
    # really does move.  So for shared-plate motions the retreat is planned
    # too and prepended to the command, making execution match the assumption.
    SHARED_PLATE = "platemid"
    # How close the acting arm's tool has to get to the shared plate, and how
    # close the idle arm has to be to it, before a retreat is worth planning.
    SHARED_PLATE_REACH = 0.20
    SHARED_PLATE_INTRUSION = 0.25

    def _shared_plate(self) -> Optional[int]:
        for body, name in self.world.body_names.items():
            if name == self.SHARED_PLATE:
                return body
        return None

    def _tool_point(self, arm, conf) -> Any:
        joints = pb_utils.get_movable_joints(arm.robot)
        saved = pb_utils.get_joint_positions(arm.robot, joints)
        try:
            pb_utils.set_joint_positions(arm.robot, joints, conf.configuration)
            return pb_utils.get_link_pose(arm.robot, arm.tool_link)[0]
        finally:
            pb_utils.set_joint_positions(arm.robot, joints, saved)

    def other_arm_states(self, arm, fluents):
        """(other arm, its conf this step, attachments) for every idle arm.

        Taken from the AtConf/Holding fluents rather than the simulator, so it
        is the configuration the plan actually leaves that arm in, not whatever
        the last stream happened to set.
        """
        confs = {args["a"]: args["q"] for args in fluents.get("AtConf", [])}
        held: Dict[int, List[Any]] = {}
        for args in fluents.get("Holding", []):
            held.setdefault(args["a"], []).append((args["b"], args["g"]))
        states = []
        for other in self.world.arms.values():
            if other.robot == arm.robot:
                continue
            states.append((other, confs.get(other.robot), held.get(other.robot, [])))
        return states

    def retreat_others(self, arm, fluents, start_conf, target_conf, key=None):
        """Body paths clearing idle arms off the shared plate, or []."""
        plate = self._shared_plate()
        if plate is None or target_conf is None:
            return [], []
        # The retreat happens before the acting arm moves, so plan it with that
        # arm at its start configuration.  Leaving it wherever the last stream
        # put it usually means both arms occupy the same hover pose, and the
        # retreat is refused for starting in collision.
        if start_conf is not None:
            start_conf.assign()
        atpose = [("atpose", a["b"], a["p"]) for a in fluents.get("AtPose", [])]
        paths, pending = [], []
        # Identifies THIS retreat: the acting arm plus the exact target conf
        # object it is heading for.  Re-sampling the same stream instance
        # reuses that object, so the retreat is recomputed; a later step has a
        # different one, so an arm parked earlier is left alone.
        key = key if key is not None else (arm.robot, id(target_conf))
        for other, conf, carried in self.other_arm_states(arm, fluents):
            if conf is None:
                continue
            note = getattr(conf, "displaced", None)
            if os.environ.get("COOKING_TRACE"):
                print(f"[retreat] act={arm.name} tgt#{id(target_conf)%10000} "
                      f"idle={other.name} conf#{id(conf)%10000} "
                      f"note={'same' if note and note['key']==key else ('other' if note else 'none')}",
                      flush=True)
            if note is not None and note["key"] != key:
                continue  # Already parked by an earlier step; nothing to redo.
            # Either fresh, or this same step being re-sampled: in both cases
            # `conf` is where the arm stands when this step begins.
            conf.assign()
            target = kuka_primitives.BodyConf(other.robot, other.q_park)
            if carried:
                body, grasp = carried[0]
                saved = pb_utils.get_pose(body)
                try:
                    result = self._retreat_hold[other.robot](
                        conf, target, body, grasp, fluents=atpose
                    )
                finally:
                    pb_utils.set_pose(body, saved)
            else:
                result = self._retreat_free[other.robot](conf, target, fluents=atpose)
            if os.environ.get("COOKING_TRACE"):
                print(f"[retreat]   rrt={'ok' if result else 'FAILED'}", flush=True)
            if result is None:
                continue  # Cannot clear it; the acting arm's plan may still fail.
            paths.extend(result[0].body_paths)
            pending.append((conf, target, key, list(result[0].body_paths)))
            pb_utils.set_joint_positions(
                other.robot, pb_utils.get_movable_joints(other.robot), other.q_park
            )
        return paths, pending

    @staticmethod
    def commit_retreat(pending) -> None:
        """Record that these arms really did move, once the plan using it holds.

        The retreat leaves an arm at q_park while its symbolic AtConf still
        names where it started, so its *next* motion has to be told to begin
        from q_park.  That is recorded on the conf object itself, which scopes
        it to this binding: IK mints a fresh BodyConf per sample, so a
        re-planned binding cannot inherit a stale record.

        Committed only after the acting arm's own trajectory is planned.  A
        retreat computed for an attempt that then fails is discarded -- leaving
        the note behind would send the idle arm's next motion off from a q_park
        it never travelled to, which is the same teleport in reverse.
        """
        for conf, target, key, paths in pending:
            conf.displaced = {"key": key, "to": target, "paths": paths}

    def retreat_for(self, key, fluents) -> List[Any]:
        """Retreat paths already committed for this action, if any.

        IK is sampled before the transit stream and is the first thing that
        needs the shared space, so it owns the decision.  The two share the
        action's ?q2 conf object, which keys the record; the transit stream
        looks it up here and carries the motion out.
        """
        paths = []
        for _other, conf, _carried in self.other_arm_states_all(fluents):
            note = getattr(conf, "displaced", None)
            if note is not None and note["key"] == key:
                paths.extend(note["paths"])
        return paths

    def other_arm_states_all(self, fluents):
        confs = {args["a"]: args["q"] for args in fluents.get("AtConf", [])}
        held: Dict[int, List[Any]] = {}
        for args in fluents.get("Holding", []):
            held.setdefault(args["a"], []).append((args["b"], args["g"]))
        return [
            (a, confs.get(a.robot), held.get(a.robot, []))
            for a in self.world.arms.values()
            if confs.get(a.robot) is not None
        ]

    @staticmethod
    def actual_conf(conf):
        """Where an arm really is, following any shared-plate retreat."""
        note = getattr(conf, "displaced", None)
        return note["to"] if note is not None else conf

    @staticmethod
    def with_retreat(retreat_paths, command):
        """Prepend the idle-arm retreat to a transit command.

        A Command is just a list of BodyPaths, each naming its own body, so one
        command can drive both arms; execute() and refine() already iterate
        them uniformly.
        """
        if not retreat_paths:
            return command
        return kuka_primitives.Command(list(retreat_paths) + list(command.body_paths))

    def assign_others(self, arm, fluents) -> None:
        """Put idle arms where the plan actually leaves them.

        The motion samplers already carry the other robot in their obstacle
        list, so this is what makes a planned trajectory honestly collision
        free.  park_others() below is the fallback used only once a retreat has
        actually been planned.
        """
        for other, conf, carried in self.other_arm_states(arm, fluents):
            if conf is None:
                continue
            self.actual_conf(conf).assign()
            for _, grasp in carried:
                grasp.assign()

    def park_others(self, arm) -> None:
        """Pin every idle arm at q_park before collision checking.

        The other arm is a static obstacle in arm.fixed_bodies, but its
        simulator configuration is whatever the previous IK or RRT left it at.
        Without this, refinement is not reproducible and a stale configuration
        of the idle arm can veto a perfectly good plan for the acting one.
        """
        for other in self.world.arms.values():
            if other.robot != arm.robot:
                pb_utils.set_joint_positions(
                    other.robot, pb_utils.get_movable_joints(other.robot), other.q_park
                )


class SamplePose(coast.Stream):
    name = "SamplePose"
    inputs = [Object("?b", "object"), Object("?s", "object")]
    outputs = [Object("?p", "pose")]
    fluents: List[str] = ["AtPose"]

    GENERIC_SAMPLE_ATTEMPTS = 200
    # Keep pot placements clear of the inner walls, and resting slightly above
    # the floor so the contact does not register as penetration.
    WALL_MARGIN = 0.005
    FLOOR_CLEARANCE = 1e-3
    PLATE_MARGIN = 0.010
    # Ingredients on a lid go into these four slots (offsets from the lid
    # centre), tried in random order.  A lid parked on the middle plate is the
    # natural staging area for a handoff, so it has to hold more than one
    # thing; four fit.
    #
    # The slots are arranged around the lid's central knob rather than in a
    # square, because the knob occupies the middle: at 0.06 from the centre a
    # 0.06-wide ingredient clears the 0.02-radius knob by 0.01, and adjacent
    # slots are 0.085 apart, which is enough to pick one without clipping its
    # neighbour.
    #
    # They also stay inside the radius within which an object blocks a top
    # grasp of the lid itself (measured on this scene: <=0.07 from the centre
    # collides, >=0.09 does not).  The knob raises the lid's grasp point above
    # the slab, so an ingredient standing on the slab is now taller than the
    # grasp point as well -- a lid holding anything is firmly ungraspable,
    # which is what enforces "a loaded lid cannot be picked up".
    LID_SLOT_R = 0.06
    LID_SLOTS = [(-LID_SLOT_R, 0.0), (LID_SLOT_R, 0.0),
                 (0.0, -LID_SLOT_R), (0.0, LID_SLOT_R)]

    def __init__(self, world: Any):
        super().__init__(world, resample_p=0.6)
        self._body_types = world.body_types
        self._fixed_bodies = world.fixed_bodies
        self._movable_bodies = world.movable_bodies
        self._robots = [arm.robot for arm in world.arms.values()]
        # Pot interiors are static geometry; compute them once per body.
        self._interior_cache: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

    # -- collision -------------------------------------------------------
    def _apply_atpose_fluents(self, fluents: Dict[str, List[Dict[str, Any]]]) -> None:
        for args in fluents.get("AtPose", []):
            pose = args.get("p")
            if pose is not None:
                pose.assign()

    def _held_bodies(self, fluents: Dict[str, List[Dict[str, Any]]]) -> set:
        """Movables that are in a gripper at this plan step.

        A held object has no AtPose fluent, so _apply_atpose_fluents leaves it
        wherever the simulator last put it -- which is the surface it was
        picked up from.  Counting it as an obstacle makes it a phantom: with
        two arms, one arm routinely holds a lid it just lifted off the very
        surface the other arm is trying to use, and every placement there is
        rejected against an object that is not actually present.
        """
        posed = {args["b"] for args in fluents.get("AtPose", [])}
        if not posed:
            return set()  # No fluent information (unit tests) -- assume nothing held.
        return {b for b in self._movable_bodies if b not in posed}

    def _in_collision(
        self,
        body: int,
        fluents: Dict[str, List[Dict[str, Any]]],
        ignore: Optional[set] = None,
    ) -> bool:
        # Evaluate collisions against the projected symbolic state, not just
        # the current simulator state.  Whether a lid is on this surface *at
        # this plan step* is precisely what most of this domain turns on.
        self._apply_atpose_fluents(fluents)
        ignore = set(ignore or set()) | self._held_bodies(fluents)
        obstacles = self._robots + self._fixed_bodies + self._movable_bodies
        for obstacle in obstacles:
            if obstacle == body or obstacle in ignore:
                continue
            if pb_utils.pairwise_collision(body, obstacle):
                return True
        return False

    # -- geometry helpers ------------------------------------------------
    def _pot_interior(self, surface: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (lower_xy, upper_xy, floor_z) of the space inside a pot.

        sample_placement() draws from the body's *outer* AABB, which spans the
        walls too, so a placement taken from there can land embedded in one --
        invisible to the collision check below, which has to ignore the support
        surface, but fatal later when IK collision-checks against the same
        walls.  `_pot_interior_bounds` derives the interior from the per-link
        AABBs instead; this only caches it, since the geometry is static.
        """
        try:
            return self._interior_cache[surface]
        except KeyError:
            pass

        interior = _pot_interior_bounds(surface)
        self._interior_cache[surface] = interior
        return interior

    def _sample_pot_interior(
        self, body: int, surface: int, fluents: Dict[str, List[Dict[str, Any]]]
    ):
        """Sample a placement resting on a pot's floor, inside its walls.

        XY is drawn with a triangular distribution peaked at the pot centre
        rather than uniformly: the wrist has to descend into a narrow opening
        to reach the object, so placements hugging a wall are usually
        unreachable even though they are collision-free.  Biasing keeps the
        whole interior reachable in principle while spending most draws where
        IK actually succeeds.
        """
        lower_xy, upper_xy, floor_z = self._pot_interior(surface)

        for _ in range(self.GENERIC_SAMPLE_ATTEMPTS):
            rotation = pb_utils.Euler(yaw=np.random.uniform(-np.pi, np.pi))
            pb_utils.set_pose(
                body, pb_utils.multiply(pb_utils.Pose(euler=rotation), pb_utils.unit_pose())
            )
            center, extent = pb_utils.get_center_extent(body)

            half = extent[:2] / 2.0 + self.WALL_MARGIN
            lo = lower_xy + half
            hi = upper_xy - half
            if np.any(lo >= hi):
                continue  # Body does not fit inside the walls at this yaw.

            xy = np.random.triangular(lo, (lo + hi) / 2.0, hi)
            z = floor_z + extent[2] / 2.0 + self.FLOOR_CLEARANCE
            point = np.array([xy[0], xy[1], z]) + (pb_utils.get_point(body) - center)
            pose = pb_utils.multiply(pb_utils.Pose(point, rotation), pb_utils.unit_pose())
            pb_utils.set_pose(body, pose)

            # The pot stays in the collision check here: unlike a flat support
            # surface, its walls -- and any lid resting on its rim -- are
            # exactly what we need to avoid.
            if not self._in_collision(body, fluents=fluents):
                return kuka_primitives.BodyPose(body, pose)

        return None

    def _sample_flat_top(
        self,
        body: int,
        surface: int,
        fluents: Dict[str, List[Dict[str, Any]]],
        margin: float,
    ):
        """Sample a stable placement on top of a flat slab (a plate)."""
        top_aabb = pb_utils.get_aabb(surface)
        lower_xy = np.array(top_aabb[0][:2], dtype=float)
        upper_xy = np.array(top_aabb[1][:2], dtype=float)
        top_z = float(top_aabb[1][2])

        for _ in range(self.GENERIC_SAMPLE_ATTEMPTS):
            rotation = pb_utils.Euler(yaw=np.random.uniform(-np.pi, np.pi))
            pb_utils.set_pose(
                body, pb_utils.multiply(pb_utils.Pose(euler=rotation), pb_utils.unit_pose())
            )
            center, extent = pb_utils.get_center_extent(body)

            half = extent[:2] / 2.0 + margin
            lo = lower_xy + half
            hi = upper_xy - half
            if np.any(lo >= hi):
                continue

            xy = np.random.triangular(lo, (lo + hi) / 2.0, hi)
            z = top_z + extent[2] / 2.0 + self.FLOOR_CLEARANCE
            point = np.array([xy[0], xy[1], z]) + (pb_utils.get_point(body) - center)
            pose = pb_utils.multiply(pb_utils.Pose(point, rotation), pb_utils.unit_pose())
            pb_utils.set_pose(body, pose)

            if not self._in_collision(body, fluents=fluents, ignore={surface}):
                return kuka_primitives.BodyPose(body, pose)

        return None

    def _hits_protrusion(self, body: int, surface: int) -> bool:
        """Collide `body` against the parts of a lid standing above its slab.

        The support surface has to be excluded from the ordinary collision
        check, because resting on it registers as contact.  That exclusion
        would also hide a lid's knob -- the very feature that stops a second
        lid lying flat on it -- so the non-base links are tested separately.
        Only meaningful for lids: a pot's non-base links are its walls, which
        a lid is supposed to rest on.
        """
        if self._body_types.get(surface) != "lid":
            return False
        for link in pb_utils.get_all_links(surface):
            if link == pb_utils.BASE_LINK:
                continue
            if pb_utils.pairwise_collision(body, (surface, [link])):
                return True
        return False

    def _support_aabb(self, surface: int):
        """The face an object placed on `surface` actually rests on.

        For a lid that is the flat slab (base_link) and NOT the whole body:
        the knob sticks up through the middle, and things placed on a lid sit
        beside it, not on top of it.  Using the whole-body AABB would float a
        second lid on top of the knob, turning an impossible stack into a
        stable-looking one.
        """
        if self._body_types.get(surface) == "lid":
            return pb_utils.get_aabb(surface, link=pb_utils.BASE_LINK)
        return pb_utils.get_aabb(surface)

    def _sample_lid_top(
        self, body: int, surface: int, fluents: Dict[str, List[Dict[str, Any]]]
    ):
        """Place an ingredient into one of the lid's four slots.

        Discrete slots rather than rejection sampling: the usable area is small
        enough that a centre-biased random draw almost always lands on top of
        whatever is already there, which showed up as ~60% of all refinement
        failures on two-ingredient instances.
        """
        top_aabb = self._support_aabb(surface)
        cx = (top_aabb[0][0] + top_aabb[1][0]) / 2.0
        cy = (top_aabb[0][1] + top_aabb[1][1]) / 2.0
        top_z = float(top_aabb[1][2])

        rotation = pb_utils.Euler(yaw=0.0)
        pb_utils.set_pose(
            body, pb_utils.multiply(pb_utils.Pose(euler=rotation), pb_utils.unit_pose())
        )
        center, extent = pb_utils.get_center_extent(body)
        z = top_z + extent[2] / 2.0 + self.FLOOR_CLEARANCE
        # Origin-to-centroid offset, captured while the body is still at the
        # measurement pose.  Reading get_point(body) inside the loop instead
        # would compound: after a rejected slot the body sits at that slot, so
        # the correction would be added on top of the previous placement.
        origin_offset = pb_utils.get_point(body) - center

        for idx in np.random.permutation(len(self.LID_SLOTS)):
            dx, dy = self.LID_SLOTS[idx]
            point = np.array([cx + dx, cy + dy, z]) + origin_offset
            pose = pb_utils.multiply(pb_utils.Pose(point, rotation), pb_utils.unit_pose())
            pb_utils.set_pose(body, pose)
            if not self._in_collision(
                body, fluents=fluents, ignore={surface}
            ) and not self._hits_protrusion(body, surface):
                return kuka_primitives.BodyPose(body, pose)
        return None

    def _centre_on_top(self, body: int, surface: int):
        """Place body centred on the top of surface with no rotation."""
        bottom_aabb = self._support_aabb(surface)
        rotation = pb_utils.Euler(yaw=0.0)
        pb_utils.set_pose(
            body, pb_utils.multiply(pb_utils.Pose(euler=rotation), pb_utils.unit_pose())
        )
        center, extent = pb_utils.get_center_extent(body)
        x = (bottom_aabb[0][0] + bottom_aabb[1][0]) / 2
        y = (bottom_aabb[0][1] + bottom_aabb[1][1]) / 2
        z = (bottom_aabb[1] + extent / 2.0)[2] + 1e-3
        point = np.array([x, y, z]) + (pb_utils.get_point(body) - center)
        pose = pb_utils.multiply(pb_utils.Pose(point, rotation), pb_utils.unit_pose())
        pb_utils.set_pose(body, pose)
        return pose

    # -- dispatch --------------------------------------------------------
    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body = inputs["b"]
        surface = inputs["s"]
        body_type = self._body_types.get(body)
        surface_type = self._body_types.get(surface)

        if body_type == "lid":
            # Pots, plates and lids all share the same 0.27x0.27 footprint, so
            # a lid can only ever be centred -- and two of them can never
            # coexist on one surface.  A lid on a plate that already holds
            # ingredients collides with them, which is what makes the
            # (deliberately unstated) "a plate must be clear to take a lid"
            # rule fail during refinement.
            #
            # Lids are allowed through as surfaces here on purpose: a lid laid
            # on another lid rests on the lower one's slab and runs into its
            # knob, so "lids do not stack" falls out of the collision check
            # below instead of being asserted by the sampler.
            if surface_type not in ("pot", "plate", "lid"):
                return None
            pose = self._centre_on_top(body, surface)
            if (
                pose is None
                or self._in_collision(body, fluents=fluents, ignore={surface})
                or self._hits_protrusion(body, surface)
            ):
                return None
            return {"p": kuka_primitives.BodyPose(body, pose)}

        if body_type == "ingredient":
            if surface_type == "pot":
                body_pose = self._sample_pot_interior(body, surface, fluents)
            elif surface_type == "plate":
                body_pose = self._sample_flat_top(
                    body, surface, fluents, margin=self.PLATE_MARGIN
                )
            elif surface_type == "lid":
                # Slot placement, and every slot is inside the radius that
                # blocks a top grasp of the lid -- so a lid holding anything
                # stays ungraspable, while still holding up to four things.
                body_pose = self._sample_lid_top(body, surface, fluents)
            else:
                return None
            return None if body_pose is None else {"p": body_pose}

        return None


class SampleGrasp(_ArmMixin, coast.Stream):
    name = "SampleGrasp"
    inputs = [Object("?a", "arm"), Object("?b", "object")]
    outputs = [Object("?g", "grasp")]
    fluents: List[str] = []

    def __init__(self, world: Any):
        super().__init__(world)
        # A BodyGrasp carries its own robot and link, so a grasp is not
        # portable between arms.
        self._sample_grasp = {
            arm.robot: kuka_primitives.get_grasp_gen(robot=arm.robot, grasp_name="top")
            for arm in world.arms.values()
        }

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        body: int = inputs["b"]
        arm = self.arm(inputs)
        if pb_utils.get_num_links(body) > 0:
            # Multi-link (the knobbed lid): the stock generator cannot handle it.
            return {"g": kuka_primitives.BodyGrasp(
                body, _top_grasp_pose(body),
                kuka_primitives.GRASP_INFO["top"].approach_pose,
                arm.robot, arm.tool_link,
            )}
        grasp = next(self._sample_grasp[arm.robot](body=body))
        return None if grasp is None else {"g": grasp[0]}


class FreeMotion(_ArmMixin, coast.Stream):
    name = "FreeMotion"
    inputs = [Object("?a", "arm"), Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?traj1", "traj")]
    # AtConf/Holding are needed to know where the OTHER arm actually is this
    # step, and what it is carrying, so a shared-plate retreat can be planned.
    fluents: List[str] = ["AtPose", "AtConf", "Holding"]

    def __init__(self, world: Any):
        super().__init__(world)
        self._sampler = {
            arm.robot: kuka_primitives.get_free_motion_gen(arm.robot, arm.fixed_bodies)
            for arm in world.arms.values()
        }

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        arm = self.arm(inputs)
        q1 = inputs["q1"]
        q2 = inputs["q2"]
        # get_free_motion_gen asserts conf1.body == conf2.body.  A symbolic
        # slip would raise AssertionError, which the planning loop does not
        # catch around stream sampling; degrade to a refinement failure.
        if q1.body != arm.robot or q2.body != arm.robot:
            return None
        q_from = self.actual_conf(q1)
        if os.environ.get("COOKING_TRACE"):
            print(f"[motion ] {arm.name} q1#{id(q1)%10000} -> from#{id(q_from)%10000}"
                  f"{'  (DISPLACED)' if q_from is not q1 else ''} tgt#{id(q2)%10000}", flush=True)
        atpose_fluents = [("atpose", args["b"], args["p"]) for args in fluents["AtPose"]]
        inherited = self.retreat_for((arm.robot, id(q2)), fluents)
        if inherited:
            # IK already decided the other arm has to move for this action.
            self.park_others(arm)
            traj = self._sampler[arm.robot](q_from, q2, fluents=atpose_fluents)
            return None if traj is None else {
                "traj1": self.with_retreat(inherited, traj[0])}
        # Otherwise plan against the other arm where it really is; if that
        # works the trajectory is honestly collision free.
        self.assign_others(arm, fluents)
        traj = self._sampler[arm.robot](q_from, q2, fluents=atpose_fluents)
        if traj is not None:
            return {"traj1": traj[0]}
        retreat, pending = self.retreat_others(arm, fluents, q_from, q2)
        if not retreat:
            return None
        self.park_others(arm)
        traj = self._sampler[arm.robot](q_from, q2, fluents=atpose_fluents)
        if traj is None:
            return None
        self.commit_retreat(pending)
        return {"traj1": self.with_retreat(retreat, traj[0])}


class HoldingMotion(_ArmMixin, coast.Stream):
    name = "HoldingMotion"
    inputs = [Object("?a", "arm"), Object("?b", "object"), Object("?g", "grasp"),
        Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?traj1", "traj")]
    fluents: List[str] = ["AtPose", "AtConf", "Holding"]

    def __init__(self, world: Any):
        super().__init__(world)
        self._sampler = {
            arm.robot: kuka_primitives.get_holding_motion_gen(arm.robot, arm.fixed_bodies)
            for arm in world.arms.values()
        }

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        arm = self.arm(inputs)
        q1 = inputs["q1"]
        q2 = inputs["q2"]
        b = inputs["b"]
        g = inputs["g"]
        if q1.body != arm.robot or q2.body != arm.robot or g.robot != arm.robot:
            return None
        q_from = self.actual_conf(q1)
        if os.environ.get("COOKING_TRACE"):
            print(f"[motion ] {arm.name} q1#{id(q1)%10000} -> from#{id(q_from)%10000}"
                  f"{'  (DISPLACED)' if q_from is not q1 else ''} tgt#{id(q2)%10000}", flush=True)
        atpose_fluents = [("atpose", args["b"], args["p"]) for args in fluents["AtPose"]]
        # The motion planner carries b via grasp.attachment(), whose assign()
        # runs on every collision check, so planning leaves b wherever the last
        # sampled configuration happened to put it -- mid-air, potentially far
        # from its placed pose.  Later streams collision-check against live
        # simulator poses, so that leftover becomes a phantom obstacle.  Put b
        # back where the caller had it.
        saved_pose = pb_utils.get_pose(b)

        def plan():
            try:
                return self._sampler[arm.robot](q_from, q2, b, g, fluents=atpose_fluents)
            finally:
                pb_utils.set_pose(b, saved_pose)

        inherited = self.retreat_for((arm.robot, id(q2)), fluents)
        if inherited:
            self.park_others(arm)
            traj = plan()
            return None if traj is None else {
                "traj1": self.with_retreat(inherited, traj[0])}
        self.assign_others(arm, fluents)
        traj = plan()
        if traj is not None:
            return {"traj1": traj[0]}
        retreat, pending = self.retreat_others(arm, fluents, q_from, q2)
        if not retreat:
            return None
        self.park_others(arm)
        traj = plan()
        if traj is None:
            return None
        self.commit_retreat(pending)
        return {"traj1": self.with_retreat(retreat, traj[0])}


class InverseKinematics(_ArmMixin, coast.Stream):
    name = "IK"
    inputs = [Object("?a", "arm"), Object("?b", "object"), Object("?p", "pose"),
        Object("?g", "grasp")]
    outputs = [Object("?q", "conf"), Object("?traj2", "traj")]
    # Unlike packing/containers, this stream declares AtPose.  The lid is this
    # domain's principal dynamic obstacle, and whether it covers a pot *at this
    # plan step* is exactly what decides whether a pick or place into that pot
    # is feasible -- so the check has to run against the projected symbolic
    # state, not against whatever the previous stream left in the simulator.
    fluents: List[str] = ["AtPose"]
    # Minimum height the approach EE pose must be above the grasp EE pose.
    APPROACH_HOVER_Z = 0.15
    # How far above a pot's rim the approach must clear before descending.
    RIM_CLEARANCE = 0.04

    def __init__(self, world: Any):
        super().__init__(world)
        self._pot_ids = [b for b, t in world.body_types.items() if t == "pot"]

    def sample(
        self, inputs: Dict[str, Any], fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        arm = self.arm(inputs)
        body: int = inputs["b"]
        body_pose: kuka_primitives.BodyPose = inputs["p"]
        grasp: kuka_primitives.BodyGrasp = inputs["g"]
        if grasp.robot != arm.robot:
            return None

        for args in fluents.get("AtPose", []):
            pose = args.get("p")
            if pose is not None:
                pose.assign()
        # ?p is an OUTPUT for place/placelid, so the target body is absent from
        # the AtPose set and has to be positioned explicitly.
        pb_utils.set_pose(body, body_pose.pose)
        self.park_others(arm)

        grasp_pose = pb_utils.end_effector_from_body(body_pose.pose, grasp.grasp_pose)

        # --- Build approach pose: hover above the grasp, EE straight down ---
        pos, quat = grasp_pose  # quat already encodes the top-grasp orientation
        approach_x, approach_y = pos[0], pos[1]
        hover_z = pos[2] + self.APPROACH_HOVER_Z

        # For pot placements, descend along the pot's own axis and start above
        # the rim.  Deliberately NOT done for plates or lids: there is no rim
        # to clear, and re-centring would tilt the descent away from vertical,
        # which is what makes a covered pot and a loaded lid infeasible.
        for pot in self._pot_ids:
            aabb = pb_utils.get_aabb(pot)
            if (
                aabb.lower[0] <= pos[0] <= aabb.upper[0]
                and aabb.lower[1] <= pos[1] <= aabb.upper[1]
            ):
                pot_point, _ = pb_utils.get_pose(pot)
                approach_x, approach_y = pot_point[0], pot_point[1]
                hover_z = max(hover_z, aabb.upper[2] + self.RIM_CLEARANCE)
                break

        approach_pose = ((approach_x, approach_y, hover_z), quat)

        q_approach = solve_arm_ik(arm, approach_pose, arm.q_home)
        if q_approach is None:
            return None

        q_grasp = solve_arm_ik(arm, grasp_pose, np.asarray(q_approach))
        if q_grasp is None:
            return None

        conf = kuka_primitives.BodyConf(arm.robot, q_approach)
        conf.assign()
        # Skip movables that are in a gripper at this step: they have no AtPose
        # fluent, so they are still sitting at the surface they were lifted
        # from and would be phantom obstacles there.  See
        # SamplePose._held_bodies.
        posed = {args["b"] for args in fluents.get("AtPose", [])}
        obstacles = list(arm.fixed_bodies) + [
            b for b in self.world.movable_bodies
            if b != body and (not posed or b in posed)
        ]
        path = pb_utils.plan_direct_joint_motion(
            arm.robot, conf.joints, q_grasp, obstacles=obstacles,
        )
        if path is None:
            return None

        traj = kuka_primitives.Command(
            [
                kuka_primitives.BodyPath(arm.robot, path),
                kuka_primitives.Attach(body, arm.robot, grasp.link),
                kuka_primitives.BodyPath(arm.robot, path[::-1], attachments=[grasp]),
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
