def generate_constraints(problem, domain, planner, pd):
    """
    Universal constraint: If Closed(lid, base) holds, block both pick and place
    for any object with that base as the support, enforcing vertical-access safety.
    """
    # Allowed symbol sets
    objects = {"egg", "bacon", "radish", "celery", "g1", "g2"}
    lids = {"lid1", "lid2", "lid3"}
    locations = {"mysink", "mystove", "mynorth"}

    # Predicates
    predicate_Closed = domain.get_predicate("closed")

    # Actions to block
    blocked_action_pick = domain.get_action("pick")
    blocked_action_place = domain.get_action("place")

    # Single universal rule: when a base is closed, block pick/place to/from that base
    for lid in lids:
        for loc in locations:
            constraint = predicate_Closed.bind([lid, loc], problem)
            for obj in objects:
                for action in (blocked_action_pick, blocked_action_place):
                    blocked_assignment = pd.Assignment()
                    blocked_assignment.assign(problem, action.parameters[0].name, obj)
                    blocked_assignment.assign(problem, action.parameters[1].name, loc)
                    planner.block_expression_action(action, blocked_assignment, constraint, pd.Assignment())


generate_constraints(problem, domain, planner, pd)