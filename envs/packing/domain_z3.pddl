(define (domain pack)
    (:requirements :strips :typing :equality :adl)
    (:types
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
    )
    (:predicates
        (holding ?obj - movable)
        (on ?obj - movable ?on - object)
        (packed ?obj - object ?box - box)
    )
    (:action pick
        :parameters (?obj - movable ?on - object)
        :precondition (and
            (on ?obj ?on)
            (forall (?o - movable) (not (holding ?o)))
            (forall (?b - box) (not (packed ?obj ?b)))
        )
        :effect (and
            (holding ?obj)
            (not (on ?obj ?on))
        )
    )
    (:action pack
        :parameters (?obj - packable ?box - box ?on - object)
        :precondition (and
            (holding ?obj)
            (packed ?on ?box)
            (forall (?l - lid) (not (on ?l ?box)))
        )
        :effect (and
            (not (holding ?obj))
            (on ?obj ?on)
            (packed ?obj ?box)
        )
    )
    (:action placelid
        :parameters (?lid - lid ?box - box)
        :precondition (and
            (holding ?lid)
            (forall (?l - lid) (not (on ?l ?box)))
        )
        :effect (and
            (on ?lid ?box)
            (not (holding ?lid))
        )
    )
)
