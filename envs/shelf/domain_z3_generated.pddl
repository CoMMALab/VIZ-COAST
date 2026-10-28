(define (domain blocks)
    (:requirements :strips :typing :equality :negative-preconditions)

    (:types
        block
        phys
    )
    
    (:constants
        b1 - block
        b2 - block
        b3 - block
        b4 - block
        b5 - block
        b6 - block
        b7 - block
        b8 - block
        b9 - block
        b10 - block
        b11 - block
        b12 - block
        loc1 - phys
        loc2 - phys
        loc3 - phys
        loc4 - phys
        loc5 - phys
        loc6 - phys
        loc7 - phys
        loc8 - phys
        loc9 - phys
        loc10 - phys
        loc11 - phys
        loc12 - phys
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
