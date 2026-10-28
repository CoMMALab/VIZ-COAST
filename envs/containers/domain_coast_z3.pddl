(define (domain pick-and-place)
    (:requirements :strips :typing :equality :adl)
    (:types
        timestep phys thing
        phys - object
        spot - phys
        thing - object
        item - thing
        lid - thing
    )
    (:constants
        mytable - phys
        mystove - spot
        mysink - spot
        mynorth - spot
        myplate - spot
        lid1 - lid
        lid2 - lid
        lid3 - lid
        g1 - item
        g2 - item
        egg - item
        bacon - item
        celery - item
        radish - item
        ;TIMESTEPS HERE

    )
    (:predicates
        (AtTimestep ?t - timestep)
        (Next ?t1 - timestep ?t2 - timestep)
        
        (Holding ?obj - thing)
        (Closed ?obj - lid ?base - spot)
        (On ?obj - thing ?on - phys)

        (Failpick ?obj - item ?on - phys ?t1 - timestep ?t2 - timestep)
        (Failplace ?obj - item ?on - phys ?t1 - timestep ?t2 - timestep)
        (Failplacetop ?obj - lid ?on - spot ?t1 - timestep ?t2 - timestep)
        (Failunplacetop ?obj - lid ?on - spot ?t1 - timestep ?t2 - timestep)

        (logpick ?obj - item ?on - phys ?t1 - timestep ?t2 - timestep)
        (logplace ?obj - item ?on - phys ?t1 - timestep ?t2 - timestep)
        (logplacetop ?obj - lid ?on - spot ?t1 - timestep ?t2 - timestep)
        (logunplacetop ?obj - lid ?on - spot ?t1 - timestep ?t2 - timestep)
    )
    (:action pick
        :parameters (?obj - item ?on - phys ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (On ?obj ?on)
            (not (Failpick ?obj ?on ?t1 ?t2))
            (forall (?o - thing) (not (Holding ?o)))
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (Holding ?obj)
            (forall (?o - phys) (not (On ?obj ?o)))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (logpick ?obj ?on ?t1 ?t2)
        )
    )
    (:action place
        :parameters (?obj - item ?on - phys ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (not (Failplace ?obj ?on ?t1 ?t2))
            (Holding ?obj)
            (AtTimestep ?t1)
            (not (= ?t1 ?t2))
            (Next ?t1 ?t2)
        )
        :effect (and
            (not (Holding ?obj))
            (On ?obj ?on)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (logplace ?obj ?on ?t1 ?t2)
        )
    )
    (:action placetop
        :parameters (?obj - lid ?base - spot ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (not (Failplacetop ?obj ?base ?t1 ?t2))
            (Holding ?obj)
            (AtTimestep ?t1)
            (not (= ?t1 ?t2))
            (Next ?t1 ?t2)
        )
        :effect (and
            (Closed ?obj ?base)
            (not (Holding ?obj))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (logplacetop ?obj ?base ?t1 ?t2)
        )
    )
    (:action unplacetop
        :parameters (?obj - lid ?base - spot ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (not (Failunplacetop ?obj ?base ?t1 ?t2))
            (Closed ?obj ?base)
            (forall (?o - thing) (not (Holding ?o)))
            (AtTimestep ?t1)
            (not (= ?t1 ?t2))
            (Next ?t1 ?t2)
        )
        :effect (and
            (Holding ?obj)
            (forall (?o - spot) (not (Closed ?obj ?o)))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (logunplacetop ?obj ?base ?t1 ?t2)
        )
    )
)