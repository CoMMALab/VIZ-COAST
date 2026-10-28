(define (domain pack)
    (:requirements :strips :typing :equality :adl)
    (:types
        timestep
        static - object
        movable - object
        box - static
        lid - movable
        packable - movable
        slab - packable
        sblock - packable
        bblock - packable
    )
    (:constants
        table - static
        box1 - box
        lid1 - lid
        slab1 - slab
        slab2 - slab
        slab3 - slab
        bb1 - bblock
        bb2 - bblock
        bb3 - bblock
        sb1 - sblock
        sb2 - sblock
        sb3 - sblock
        ;timesteps here
    )
    (:predicates
        (attimestep ?t - timestep)
        (next ?t1 - timestep ?t2 - timestep)

        (holding ?obj - movable)
        (on ?obj - movable ?on - object)
        (packed ?obj - object ?box - box)

        (Failpick ?obj - movable ?on - object ?t1 - timestep ?t2 - timestep)
        (Failpack ?obj - packable ?box - box ?on - object ?t1 - timestep ?t2 - timestep)
        (Failplacelid ?lid - lid ?box - box ?t1 - timestep ?t2 - timestep)

        (logpick ?obj - movable ?on - object ?t1 - timestep ?t2 - timestep)
        (logpack ?obj - packable ?box - box ?on - object ?t1 - timestep ?t2 - timestep)
        (logplacelid ?lid - lid ?box - box ?t1 - timestep ?t2 - timestep)
    )
    (:action pick
        :parameters (?obj - movable ?on - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (on ?obj ?on)
            (not (failpick ?obj ?on ?t1 ?t2))
            (forall (?o - movable) (not (holding ?o)))
            (forall (?b - box) (not (packed ?obj ?b)))
            (not (= ?t1 ?t2))
            (attimestep ?t1)
            (next ?t1 ?t2)
        )
        :effect (and
            (holding ?obj)
            (not (on ?obj ?on))
            (not (attimestep ?t1))
            (attimestep ?t2)
            (logpick ?obj ?on ?t1 ?t2)
        )
    )
    (:action pack
        :parameters (?obj - packable ?box - box ?on - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (holding ?obj)
            (not (failpack ?obj ?box ?on ?t1 ?t2))
            (packed ?on ?box)
            (forall (?l - lid) (not (on ?l ?box)))
            (not (= ?t1 ?t2))
            (attimestep ?t1)
            (next ?t1 ?t2)
        )
        :effect (and
            (not (holding ?obj))
            (on ?obj ?on)
            (packed ?obj ?box)
            (not (attimestep ?t1))
            (attimestep ?t2)
            (logpack ?obj ?box ?on ?t1 ?t2)
        )
    )
    (:action placelid
        :parameters (?lid - lid ?box - box ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (holding ?lid)
            (not (failplacelid ?lid ?box ?t1 ?t2))
            (forall (?l - lid) (not (on ?l ?box)))
            (not (= ?t1 ?t2))
            (attimestep ?t1)
            (next ?t1 ?t2)
        )
        :effect (and
            (on ?lid ?box)
            (not (holding ?lid))
            (not (attimestep ?t1))
            (attimestep ?t2)
            (logplacelid ?lid ?box ?t1 ?t2)
        )
    )
)
