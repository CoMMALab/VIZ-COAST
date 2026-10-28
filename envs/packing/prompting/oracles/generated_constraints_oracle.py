def generate_constraints(problem, domain, planner, pd):
    slabs = {"slab1", "slab2", "slab3"}
    bblocks = {"bb1", "bb2", "bb3"}
    sblocks = {"sb1", "sb2", "sb3"}
    boxes = {"box1"}

    predicate_on = domain.get_predicate("on")
    predicate_packed = domain.get_predicate("packed")

    action_pack = domain.get_action("pack")

    all_objects = slabs | bblocks | sblocks

    for box in boxes:
        # 1. Kinematically infeasible: Larger objects on smaller ones
        for slab in slabs:
            for smaller in bblocks | sblocks:
                constraint = predicate_packed.bind([smaller, box], problem)
                blocked_assignment = pd.Assignment()
                for o, param in zip([slab, box, smaller], action_pack.parameters):
                    blocked_assignment.assign(problem, param.name, o)
                planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        for bb in bblocks:
            for sb in sblocks:
                constraint = predicate_packed.bind([sb, box], problem)
                blocked_assignment = pd.Assignment()
                for o, param in zip([bb, box, sb], action_pack.parameters):
                    blocked_assignment.assign(problem, param.name, o)
                planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        # 2. Assigning multiple objects to a surface that lacks sufficient area
        for slab in slabs:
            for obj in all_objects:
                if slab != obj:
                    constraint = predicate_on.bind([slab, box], problem)
                    blocked_assignment = pd.Assignment()
                    for o, param in zip([obj, box, box], action_pack.parameters):
                        blocked_assignment.assign(problem, param.name, o)
                    planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

                    constraint = predicate_on.bind([obj, box], problem)
                    blocked_assignment = pd.Assignment()
                    for o, param in zip([slab, box, box], action_pack.parameters):
                        blocked_assignment.assign(problem, param.name, o)
                    planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        for slab in slabs:
            for slab_on in slabs:
                if slab != slab_on:
                    for obj in all_objects:
                        if slab_on != obj:
                            constraint = predicate_on.bind([slab_on, slab], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([obj, box, slab], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

                            constraint = predicate_on.bind([obj, slab], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([slab_on, box, slab], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        for bb in bblocks:
            for bb_on in bblocks:
                if bb != bb_on:
                    for obj in bblocks | sblocks:
                        if bb_on != obj:
                            constraint = predicate_on.bind([bb_on, bb], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([obj, box, bb], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

                            constraint = predicate_on.bind([obj, bb], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([bb_on, box, bb], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        for sb in sblocks:
            for sb_on in sblocks:
                if sb != sb_on:
                    for obj in sblocks:
                        if sb_on != obj:
                            constraint = predicate_on.bind([sb_on, sb], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([obj, box, sb], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

                            constraint = predicate_on.bind([obj, sb], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([sb_on, box, sb], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        # 3. Height constraints (exceeding 0.20 internal box height)
        for bb_top in bblocks:
            for bb_mid in bblocks:
                if bb_top != bb_mid:
                    for bb_bot in bblocks:
                        if bb_mid != bb_bot and bb_top != bb_bot:
                            constraint = predicate_on.bind([bb_mid, bb_bot], problem)
                            blocked_assignment = pd.Assignment()
                            for o, param in zip([bb_top, box, bb_mid], action_pack.parameters):
                                blocked_assignment.assign(problem, param.name, o)
                            planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

                    for slab_bot in slabs:
                        constraint = predicate_on.bind([bb_mid, slab_bot], problem)
                        blocked_assignment = pd.Assignment()
                        for o, param in zip([bb_top, box, bb_mid], action_pack.parameters):
                            blocked_assignment.assign(problem, param.name, o)
                        planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())

        for sb_top in sblocks:
            for bb_mid in bblocks:
                for bb_bot in bblocks:
                    if bb_mid != bb_bot:
                        constraint = predicate_on.bind([bb_mid, bb_bot], problem)
                        blocked_assignment = pd.Assignment()
                        for o, param in zip([sb_top, box, bb_mid], action_pack.parameters):
                            blocked_assignment.assign(problem, param.name, o)
                        planner.block_expression_action(action_pack, blocked_assignment, constraint, pd.Assignment())
generate_constraints(problem, domain, planner, pd)