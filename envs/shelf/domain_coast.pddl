(define (domain blocks)
    (:requirements :strips :typing :equality :negative-preconditions :conditional-effects)

    (:types
        block
        phys
    )

    (:constants
        ; Filled in at runtime by generate_domain_pddl() in envs/shelf/run.py,
        ; sized to the current grid (b1..bN, loc1..locN). See
        ; domain_coast_generated.pddl for the file actually used to plan.
    )

    (:predicates
        (ontable ?b - block ?loc - phys)
        (handempty)
        (clear ?a - phys)
        (holding ?b - block)
        (Failpick ?b - block ?loc - phys)
        (Failplace ?b - block ?loc - phys)
        (logpick ?b - block ?loc - phys)
        (logplace ?b - block ?loc - phys)
    )

    (:action pick
        :parameters (?b - block ?loc - phys)
        :precondition (and
                        (not (Failpick ?b ?loc))
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
                        (not (Failplace ?b ?loc))
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
