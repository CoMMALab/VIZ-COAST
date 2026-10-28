(define (domain pick-and-place)
    (:requirements :strips :typing :equality :negative-preconditions :conditional-effects)
    (:types
        timestep
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
        t1 - timestep
        t2 - timestep
        t3 - timestep
        t4 - timestep
        t5 - timestep
        t6 - timestep
        t7 - timestep
        t8 - timestep
        t9 - timestep
        t10 - timestep
        t11 - timestep
        t12 - timestep
        t13 - timestep
        t14 - timestep
        t15 - timestep
        t16 - timestep
        t17 - timestep
        t18 - timestep
        t19 - timestep
        t20 - timestep
        t21 - timestep
        t22 - timestep
        t23 - timestep
        t24 - timestep
        t25 - timestep
        t26 - timestep
        t27 - timestep
        t28 - timestep
        t29 - timestep
        t30 - timestep
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
