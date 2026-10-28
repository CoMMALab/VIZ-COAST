from typing import List, Optional, NamedTuple, Set

import os
from os.path import exists
import subprocess

import symbolic


class PlannerNode(NamedTuple):
    action: str
    state: Set[str]


def get_lama(**kwargs):
    return [
        "--if-unit-cost",
        "--evaluator",
        "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),pref={pref})".format(**kwargs),
        "--evaluator",
        "hff=ff()",
        "--search",
        """iterated([
                         lazy_greedy([hff,hlm],preferred=[hff,hlm]),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=5),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=3),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=2),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=1)
                         ],repeat_last=true,continue_on_fail=true,bound=16)""",
        "--if-non-unit-cost",
        "--evaluator",
        "hlm1=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref={pref})".format(
            **kwargs
        ),
        "--evaluator",
        "hff1=ff(transform=adapt_costs(one))",
        "--evaluator",
        "hlm2=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(plusone),pref={pref})".format(
            **kwargs
        ),
        "--evaluator",
        "hff2=ff(transform=adapt_costs(plusone))",
        "--search",
        """iterated([
                         lazy_greedy([hff1,hlm1],preferred=[hff1,hlm1], cost_type=one,reopen_closed=false),
                         lazy_greedy([hff2,hlm2],preferred=[hff2,hlm2], reopen_closed=false),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=5),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=3),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=2),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=1)
                         ],repeat_last=true,continue_on_fail=true,bound=16),""",
        # Append --always to be on the safe side if we want to append
        # additional options later.
        "--always",
    ]
def get_ff(task_horizon=50):
    return [
        '--evaluator',
         "h=ff(transform=adapt_costs(cost_type=PLUSONE))",
            '--search',
            f"eager(alt([single(sum([g(), weight(h,2)])), single(sum([g(),weight(h,2)]),pref_only=true)]), preferred=[h],cost_type=PLUSONE,bound={task_horizon})"
    ]

def get_lama_first():
    return [
        "--evaluator",
        "hlm=lmcount(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false)",
        "--evaluator",
        "hff=ff(transform=adapt_costs(one))",
        "--search",
        """lazy_greedy([hff,hlm],preferred=[hff,hlm],
                               cost_type=one,reopen_closed=false, bound=50)""",
    ]


def get_seq_sat_fd_autotune_1():
    return [
        "--evaluator",
        "hff=ff(transform=adapt_costs(one))",
        "--evaluator",
        "hcea=cea()",
        "--evaluator",
        "hcg=cg(transform=adapt_costs(plusone))",
        "--evaluator",
        "hgc=goalcount()",
        "--evaluator",
        "hAdd=add()",
        "--search",
        """iterated([
lazy(alt([single(sum([g(),weight(hff,10)])),
          single(sum([g(),weight(hff,10)]),pref_only=true)],
         boost=2000),
     preferred=[hff],reopen_closed=false,cost_type=one),
lazy(alt([single(sum([g(),weight(hAdd,7)])),
          single(sum([g(),weight(hAdd,7)]),pref_only=true),
          single(sum([g(),weight(hcg,7)])),
          single(sum([g(),weight(hcg,7)]),pref_only=true),
          single(sum([g(),weight(hcea,7)])),
          single(sum([g(),weight(hcea,7)]),pref_only=true),
          single(sum([g(),weight(hgc,7)])),
          single(sum([g(),weight(hgc,7)]),pref_only=true)],
         boost=1000),
     preferred=[hcea,hgc],reopen_closed=false,cost_type=one),
lazy(alt([tiebreaking([sum([g(),weight(hAdd,3)]),hAdd]),
          tiebreaking([sum([g(),weight(hAdd,3)]),hAdd],pref_only=true),
          tiebreaking([sum([g(),weight(hcg,3)]),hcg]),
          tiebreaking([sum([g(),weight(hcg,3)]),hcg],pref_only=true),
          tiebreaking([sum([g(),weight(hcea,3)]),hcea]),
          tiebreaking([sum([g(),weight(hcea,3)]),hcea],pref_only=true),
          tiebreaking([sum([g(),weight(hgc,3)]),hgc]),
          tiebreaking([sum([g(),weight(hgc,3)]),hgc],pref_only=true)],
         boost=5000),
     preferred=[hcea,hgc],reopen_closed=false,cost_type=normal),
eager(alt([tiebreaking([sum([g(),weight(hAdd,10)]),hAdd]),
           tiebreaking([sum([g(),weight(hAdd,10)]),hAdd],pref_only=true),
           tiebreaking([sum([g(),weight(hcg,10)]),hcg]),
           tiebreaking([sum([g(),weight(hcg,10)]),hcg],pref_only=true),
           tiebreaking([sum([g(),weight(hcea,10)]),hcea]),
           tiebreaking([sum([g(),weight(hcea,10)]),hcea],pref_only=true),
           tiebreaking([sum([g(),weight(hgc,10)]),hgc]),
           tiebreaking([sum([g(),weight(hgc,10)]),hgc],pref_only=true)],
          boost=500),
      preferred=[hcea,hgc],reopen_closed=true,cost_type=normal, bound=25)
],repeat_last=true,continue_on_fail=true)""",
    ]


def task_plan(
    log, exp: str, problem_state: str, pddl: symbolic.Pddl, task_horizon: int = 50
) -> Optional[List[PlannerNode]]:
# ) -> Optional[List[symbolic.PlannerNode]]:

    path_to_problem = f"envs/{exp}/temp_problem.pddl"
    path_to_plan = f"envs/{exp}/fd_plan.txt"
    path_to_sas = f"envs/{exp}/output.sas"
    print("[fdplan.task_plan] domain: ", pddl.domain_pddl)

    # Remove stale intermediate files to avoid corruption across iterations.
    for stale in (path_to_plan, path_to_sas, "output.sas"):
        if exists(stale):
            os.remove(stale)

    with open(path_to_problem, "w") as f:
        f.write(problem_state)

    standard_config = [
        "python3",
        "external/downward/fast-downward.py",
        "--plan-file",
        path_to_plan,
        "--sas-file",
        path_to_sas,
        "--keep-sas-file",
        "--log-level",
        "warning",
        pddl.domain_pddl,
        path_to_problem,
    ]
            
    # standard_config.extend(get_lama(pref="true"))
    # standard_config.extend(get_lama_first())
    standard_config.extend(get_ff(task_horizon=task_horizon))
    # standard_config.extend(get_seq_sat_fd_autotune_1())

    print(f"[fdplan.task_plan] Running Fast Downward with domain={pddl.domain_pddl}")
    with log.profile("planning"):
        result = subprocess.run(
            standard_config,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Print FD output so translator/search errors are visible.
            if result.stdout:
                print("[fdplan] FD stdout:", result.stdout[-2000:])
            if result.stderr:
                print("[fdplan] FD stderr:", result.stderr[-2000:])

    # Clean up the SAS file after use.
    if exists(path_to_sas):
        os.remove(path_to_sas)

    if not exists(path_to_plan):
        print("[fdplan.task_plan] Fast Downward found no plan")
        return None

    with open(path_to_plan, "r") as f:
        actions = f.readlines()

    state = pddl.initial_state
    plan = [PlannerNode("", state)]
    for a in actions[:-1]:
        all_symbols = a.strip()[1:-1].split(" ")
        a = f"{all_symbols[0]}({','.join(all_symbols[1:])})"
        state = pddl.next_state(state, a)
        plan.append(PlannerNode(a, state))

    print(f"[fdplan.task_plan] Found plan with {len(plan) - 1} action(s)")
    return plan
