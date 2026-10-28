(define (domain cooking)
    (:requirements :strips :typing :equality :adl)
    (:types
        timestep
        arm - object
        static - object
        movable - object
        surface - static
        pot - surface
        plate - surface
        ingredient - movable
        lid - movable
    )
    (:constants
        table - static
        left - arm
        right - arm
        frypot - pot
        steampot - pot
        plateleft - plate
        plateright - plate
        platemid - plate
        lidleft - lid
        lidright - lid
        ing1 - ingredient
        ing2 - ingredient
        ing3 - ingredient
        ing4 - ingredient
        ing5 - ingredient
        ing6 - ingredient
        ing7 - ingredient
        ing8 - ingredient
    )
    (:predicates
        (holding ?a - arm ?obj - movable)
        (on ?obj - movable ?on - object)
        ; Well-formedness only: marks the objects an item may rest on (pots,
        ; plates, lids).  Static, asserted in :init.  This is NOT one of the
        ; physical rules the domain deliberately omits.
        (support ?x - object)
        ; Well-formedness only: true while an object is in a gripper, so that
        ; nothing can be set down onto an object that is currently being
        ; carried (which includes setting a lid down onto itself).
        (held ?x - object)
        (isfryer ?x - object)
        (issteamer ?x - object)
        (uncooked ?i - ingredient)
        (fried ?i - ingredient)
        (steamed ?i - ingredient)
    )
    (:action pick
        :parameters (?a - arm ?obj - movable ?on - object)
        :precondition (and
            (on ?obj ?on)
            (forall (?o - movable) (not (holding ?a ?o)))
        )
        :effect (and
            (holding ?a ?obj)
            (held ?obj)
            (not (on ?obj ?on))
        )
    )
    ; Putting an ingredient into the frying pot fries it.
    (:action place
        :parameters (?a - arm ?obj - ingredient ?on - object)
        :precondition (and
            (holding ?a ?obj)
            (support ?on)
            (not (held ?on))
        )
        :effect (and
            (not (holding ?a ?obj))
            (not (held ?obj))
            (on ?obj ?on)
            (when (isfryer ?on) (and (fried ?obj) (not (uncooked ?obj))))
        )
    )
    ; Covering the steaming pot steams everything that is inside it.
    (:action placelid
        :parameters (?a - arm ?l - lid ?on - object)
        :precondition (and
            (holding ?a ?l)
            (support ?on)
            (not (held ?on))
        )
        :effect (and
            (not (holding ?a ?l))
            (not (held ?l))
            (on ?l ?on)
            (forall (?i - ingredient)
                (when (and (issteamer ?on) (on ?i ?on))
                    (and (steamed ?i) (not (uncooked ?i)))))
        )
    )
)
