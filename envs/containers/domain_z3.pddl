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
        myplate - spot
        mystove - spot
        mysink - spot
        mynorth - spot
        lid1 - lid
        lid2 - lid
        lid3 - lid
        g1 - item
        g2 - item
        egg - item
        bacon - item
        celery - item
        radish - item
    )
    (:predicates
        (holding ?obj - thing)
        (on ?obj - thing ?on - phys)
        (closed ?obj - lid ?base - spot)
    )
    (:action pick
        :parameters (?obj - item ?on - phys)
        :precondition (and
            (on ?obj ?on)
            (forall (?o - thing) (not (holding ?o)))
        )
        :effect (and
            (holding ?obj)
            (forall (?on - phys) (not (on ?obj ?on)))
        )
    )
    (:action place
        :parameters (?obj - item ?on - phys)
        :precondition (and
            (holding ?obj)
        )
        :effect (and
            (not (holding ?obj))
            (on ?obj ?on)
        )
    )
    (:action placetop
        :parameters (?obj - lid ?base - spot)
        :precondition (and
            (holding ?obj)
            (forall (?o - lid) (not (closed ?o ?base)))
        )
        :effect (and
            (closed ?obj ?base)
            (not (holding ?obj))
        )
    )
    (:action unplacetop
        :parameters (?obj - lid ?base - spot)
        :precondition (and
            (closed ?obj ?base)
            (forall (?o - thing) (not (holding ?o)))
        )
        :effect (and
            (holding ?obj)
            (forall (?on - spot) (not (closed ?obj ?on)))
        )
    )
)