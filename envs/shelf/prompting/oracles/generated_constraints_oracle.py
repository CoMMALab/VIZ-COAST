def generate_constraints(problem, domain, planner, pd):

    # References to maximum possible instances of objects in domain
    blocks = {"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12"}
    locations = {"loc1", "loc2", "loc3", "loc4", "loc5", "loc6", "loc7", "loc8", "loc9", "loc10", "loc11", "loc12"}

    # References to all predicates in domain
    predicate_clear = domain.get_predicate("clear")
    predicate_ontable = domain.get_predicate("ontable")
    predicate_holding = domain.get_predicate("holding")
    predicate_handempty = domain.get_predicate("handempty")

    # Declare only actions we may want to block
    action_pick = domain.get_action("pick")
    action_place = domain.get_action("place")

    # Mapping of locations to the locations that block their access from negative Y direction
    # (locations in the same column with smaller Y coordinates)
    blocking_map = {
        "loc1": ["loc2", "loc3", "loc4"],
        "loc2": ["loc3", "loc4"],
        "loc3": ["loc4"],
        "loc5": ["loc6", "loc7", "loc8"],
        "loc6": ["loc7", "loc8"],
        "loc7": ["loc8"],
        "loc9": ["loc10", "loc11", "loc12"],
        "loc10": ["loc11", "loc12"],
        "loc11": ["loc12"]
    }

    for b in blocks:
        for loc, obstacles in blocking_map.items():
            for obs in obstacles:
                constraint = pd.make_not(predicate_clear.bind([obs], problem))
                
                # Block pick action
                blocked_assignment_pick = pd.Assignment()
                for obj, param in zip([b, loc], action_pick.parameters):
                    blocked_assignment_pick.assign(problem, param.name, obj)
                planner.block_expression_action(action_pick, blocked_assignment_pick, constraint, pd.Assignment())
                
                # Block place action
                blocked_assignment_place = pd.Assignment()
                for obj, param in zip([b, loc], action_place.parameters):
                    blocked_assignment_place.assign(problem, param.name, obj)
                planner.block_expression_action(action_place, blocked_assignment_place, constraint, pd.Assignment())

generate_constraints(problem, domain, planner, pd)