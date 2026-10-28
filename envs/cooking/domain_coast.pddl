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
        ;timesteps here
    )
    (:predicates
        (attimestep ?t - timestep)
        (next ?t1 - timestep ?t2 - timestep)

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

        ; Fail<action> / log<action> arg lists must mirror the action's own
        ; :parameters in the same order, with ?t1 ?t2 appended -- coast's
        ; constraint.py binds them positionally and lowercases the head.
        (Failpick ?a - arm ?obj - movable ?on - object ?t1 - timestep ?t2 - timestep)
        (Failplace ?a - arm ?obj - ingredient ?on - object ?t1 - timestep ?t2 - timestep)
        (Failplacelid ?a - arm ?l - lid ?on - object ?t1 - timestep ?t2 - timestep)

        (logpick ?a - arm ?obj - movable ?on - object ?t1 - timestep ?t2 - timestep)
        (logplace ?a - arm ?obj - ingredient ?on - object ?t1 - timestep ?t2 - timestep)
        (logplacelid ?a - arm ?l - lid ?on - object ?t1 - timestep ?t2 - timestep)
    )
    (:action pick
        :parameters (?a - arm ?obj - movable ?on - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (on ?obj ?on)
            (not (failpick ?a ?obj ?on ?t1 ?t2))
            (forall (?o - movable) (not (holding ?a ?o)))
            (not (= ?t1 ?t2))
            (attimestep ?t1)
            (next ?t1 ?t2)
        )
        :effect (and
            (holding ?a ?obj)
            (held ?obj)
            (not (on ?obj ?on))
            (not (attimestep ?t1))
            (attimestep ?t2)
            (logpick ?a ?obj ?on ?t1 ?t2)
        )
    )
    ; Putting an ingredient into the frying pot fries it.
    (:action place
        :parameters (?a - arm ?obj - ingredient ?on - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (holding ?a ?obj)
            (support ?on)
            (not (held ?on))
            (not (failplace ?a ?obj ?on ?t1 ?t2))
            (not (= ?t1 ?t2))
            (attimestep ?t1)
            (next ?t1 ?t2)
        )
        :effect (and
            (not (holding ?a ?obj))
            (not (held ?obj))
            (on ?obj ?on)
            (when (isfryer ?on) (and (fried ?obj) (not (uncooked ?obj))))
            (not (attimestep ?t1))
            (attimestep ?t2)
            (logplace ?a ?obj ?on ?t1 ?t2)
        )
    )
    ; Covering the steaming pot steams everything that is inside it.
    (:action placelid
        :parameters (?a - arm ?l - lid ?on - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (holding ?a ?l)
            (support ?on)
            (not (held ?on))
            (not (failplacelid ?a ?l ?on ?t1 ?t2))
            (not (= ?t1 ?t2))
            (attimestep ?t1)
            (next ?t1 ?t2)
        )
        :effect (and
            (not (holding ?a ?l))
            (not (held ?l))
            (on ?l ?on)
            (forall (?i - ingredient)
                (when (and (issteamer ?on) (on ?i ?on))
                    (and (steamed ?i) (not (uncooked ?i)))))
            (not (attimestep ?t1))
            (attimestep ?t2)
            (logplacelid ?a ?l ?on ?t1 ?t2)
        )
    )
)
