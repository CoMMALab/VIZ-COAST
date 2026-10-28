def generate_constraints(problem, domain, planner, pd):
    """Ground-truth constraints for the cooking domain.

    The PDDL domain is deliberately naive: it knows nothing about which arm can
    reach what, nor about lids occluding the things they cover.  These are the
    rules that downward refinement discovers geometrically.

    Two different mechanisms are used, and the choice matters:

    * `planner.add_constraint(OVER_ALL, expr, False)` forbids a *state* at every
      step.  Use it whenever the rule is a property of the world rather than of
      an action -- "two lids never share a surface" is true no matter which arm
      put them there, or in what order.
    * `planner.block_expression_action(...)` forbids an *action while an
      expression holds*.  Use it when the rule really is about acting.  Note
      that it is evaluated per horizon step, and a step can hold more than one
      action, so two actions in the same step both see the state *before*
      either fired.  Stating a genuinely stateful rule this way leaks: blocking
      "put a lid on a plate that already has one" does not stop both arms
      putting a lid there simultaneously.  That is why rules 3 and 4 below are
      invariants, not action blocks.
    """
    arms = ["left", "right"]
    ingredients = [f"ing{i + 1}" for i in range(8)]
    lids = ["lidleft", "lidright"]
    pots = ["frypot", "steampot"]
    plates = ["plateleft", "plateright", "platemid"]
    surfaces = pots + plates

    # Each arm can only reach its own pot and plate, plus the shared middle plate.
    unreachable = {
        "left": ["steampot", "plateright"],
        "right": ["frypot", "plateleft"],
    }

    predicate_on = domain.get_predicate("on")
    action_pick = domain.get_action("pick")
    action_place = domain.get_action("place")
    action_placelid = domain.get_action("placelid")

    def block(action, args, constraint):
        blocked_assignment = pd.Assignment()
        for obj, param in zip(args, action.parameters):
            blocked_assignment.assign(problem, param.name, obj)
        planner.block_expression_action(
            action, blocked_assignment, constraint, pd.Assignment()
        )

    def forbid_state(*props):
        """Forbid the conjunction of these (predicate, args) pairs, at all times."""
        exprs = [predicate_on.bind(list(args), problem) for args in props]
        planner.add_constraint(
            pd.Planner.ConstraintTime.OVER_ALL, pd.make_and(exprs), False
        )

    def always():
        x = predicate_on.bind(["ing1", "platemid"], problem)
        return pd.make_or([x, pd.make_not(x)])

    # 1. Reachability: an arm cannot touch anything on the other arm's side.
    #    Unconditional, so no leak is possible.
    for arm in arms:
        for surface in unreachable[arm]:
            for obj in ingredients:
                block(action_pick, [arm, obj, surface], always())
                block(action_place, [arm, obj, surface], always())
            for lid in lids:
                block(action_pick, [arm, lid, surface], always())
                block(action_placelid, [arm, lid, surface], always())
            # Anything resting on a lid that is itself on an unreachable
            # surface is equally out of range.
            for lid in lids:
                constraint = predicate_on.bind([lid, surface], problem)
                for obj in ingredients:
                    block(action_pick, [arm, obj, lid], constraint)
                    block(action_place, [arm, obj, lid], constraint)

    # 2. A lid on a pot blocks access to everything inside it.  Genuinely about
    #    acting -- a lidded pot with food in it is a legal, and for steaming a
    #    required, state -- so this stays an action block.
    for pot in pots:
        for lid in lids:
            constraint = predicate_on.bind([lid, pot], problem)
            for arm in arms:
                for obj in ingredients:
                    block(action_pick, [arm, obj, pot], constraint)
                    block(action_place, [arm, obj, pot], constraint)

    # 3. Two lids never share a surface, and never stack: pot, plate and lid
    #    all have the same footprint, and each lid carries a central knob that
    #    a second lid would have to pass through.  State properties ->
    #    invariants.
    #
    #    Note the knob is a real feature of the model, not an assumption made
    #    here: refinement rejects a lid-on-lid placement on its own.  This rule
    #    exists so the planner stops *proposing* it -- without it the search
    #    re-derives the same rejection hundreds of times per run.
    for surface in surfaces:
        forbid_state((lids[0], surface), (lids[1], surface))
    for lid in lids:
        for other in lids:
            if other == lid:
                continue
            forbid_state((lid, other))

    # 4. A lid covers its whole plate, so a plate never holds a lid and an
    #    ingredient at once -- in either order.  A state property -> invariant.
    #    (Pots are exempt: their contents sit below the rim, which is the whole
    #    point of lidding the steamer.)
    for plate in plates:
        for lid in lids:
            for obj in ingredients:
                forbid_state((obj, plate), (lid, plate))

    # 5. A lid carrying ingredients cannot be grasped -- moving the lid risks 
    # spilling the contents.
    for lid in lids:
        for obj in ingredients:
            constraint = predicate_on.bind([obj, lid], problem)
            for arm in arms:
                for surface in surfaces:
                    block(action_pick, [arm, lid, surface], constraint)


generate_constraints(problem, domain, planner, pd)
