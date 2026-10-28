(define (domain blocks)
    (:requirements :strips :typing :equality :negative-preconditions)

    (:types
        block
        phys
    )
    
    (:constants
        ; Filled in at runtime by generate_domain_pddl() in envs/shelf/run.py,
        ; sized to the current grid (b1..bN, loc1..locN). See
        ; domain_z3_generated.pddl for the file actually used to plan.
    )

    (:predicates
        (ontable ?b - block ?loc - phys)
        (handempty)
        (clear ?a - phys)
        (holding ?b - block)
    )

    (:action pick
        :parameters (?b - block ?loc - phys)
        :precondition (and
                        (ontable ?b ?loc)
                        (handempty)
                        (not (clear ?loc))
                    )
        :effect (and
                    (not (ontable ?b ?loc))
                    (not (handempty))
                    (holding ?b)
                    (clear ?loc)
                )
    )
    (:action place
        :parameters (?b - block ?loc - phys)
        :precondition (and
                        (holding ?b)
                        (clear ?loc)
                    )
        :effect (and
                    (not (holding ?b))
                    (ontable ?b ?loc)
                    (not (clear ?loc))
                    (handempty)
                )
    )
)
