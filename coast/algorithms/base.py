import abc
import collections
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Type, Union

import symbolic  # type: ignore

from coast import utils
from coast.action import Action, ActionCall
from coast.constraint import Constraint
from coast.object import Object
from coast.stream import Stream, StreamConstraint, StreamInstance
from coast.formula import _P


class Log(utils.Profiler):
    def __init__(self):
        super().__init__()
        self._stream_profiler = utils.Profiler()
        self._successes: Dict[str, List[bool]] = collections.defaultdict(list)
        self._iterations = 0

        self._num_successes: Dict[str, int] = collections.defaultdict(int)
        self._num_samples: Dict[str, int] = collections.defaultdict(int)
        self.num_fail_task_plans = 0
        self.num_success_task_plans = 0
        self.num_prompts = 0  # how many times the VLM was actually called
    
    def _get_key(self, stream_instance: StreamInstance) -> str:
        return stream_instance.stream.name
        return f"{stream_instance.stream.name}({', '.join(arg.name for arg in stream_instance.inputs.values())})"

    def profile_stream(
        self, stream_instance: StreamInstance
    ) -> utils.Profiler.ProfilerContext:
        return self._stream_profiler.profile(self._get_key(stream_instance))
    
    def log(
        self,
        stream_instance: StreamInstance,
        maybe_bindings: Optional[Union[Mapping[str, Object], StreamConstraint]],
    ) -> None:
        key = self._get_key(stream_instance)
        is_success = maybe_bindings is not None and not isinstance(
            maybe_bindings, StreamConstraint
        )
        self._successes[key].append(is_success)

        if is_success:
            self._num_successes[key] += 1
        self._num_samples[key] += 1

    def set_iteration(self, idx_iter: int) -> None:
        self._iterations = max(self._iterations, idx_iter)

    def _success_rate(self, key: str) -> float:
        if key not in self._num_successes:
            return 1.0
        return self._num_successes[key] / self._num_samples[key]

    def success_rate(self, stream_instance: StreamInstance) -> float:
        return self._success_rate(self._get_key(stream_instance))

    def _total_time(self, key: str) -> float:
        return self._stream_profiler.compute_sum(key)

    def total_time(self, stream_instance: StreamInstance) -> float:
        return self._total_time(self._get_key(stream_instance))

    def _average_time(self, key: str) -> float:
        if key not in self._stream_profiler._tictocs:
            return 0.0
        return self._stream_profiler.compute_average(key)

    def average_time(self, stream_instance: StreamInstance) -> float:
        return self._average_time(self._get_key(stream_instance))

    # Keys whose sum constitutes the true wall-clock total.
    # prompting and planning are nested inside search, so they are
    # reported separately but NOT double-counted into total.
    _TOTAL_KEYS = ("search", "retrace", "process")

    def print(self) -> None:
        print("===== Log =====")
        for key in self._stream_profiler._tictocs:
            print(
                f"  {key} [{self._num_successes[key]} / {self._num_samples[key]} succeeded]: {self._stream_profiler.compute_sum(key)}s"
            )
        print()
        for key in self._tictocs:
            t_key = self.compute_sum(key)
            print(f"  {key}: {t_key}s")
        t_total = sum(self.compute_sum(k) for k in self._TOTAL_KEYS if k in self._tictocs)
        print(f"  total: {t_total}s, {self._iterations} iterations, {self.num_prompts} prompt(s)")
        print()

    def get_timing(self) -> Dict[str, float]:
        res: Dict[str, float] = {}
        for key in self._tictocs:
            t_key = self.compute_sum(key)
            print(f"  {key}: {t_key}s")
            res[key] = t_key
        # Total = search + retrace + process (no double-counting of
        # prompting/planning which are nested inside search).
        t_total = sum(res.get(k, 0.0) for k in self._TOTAL_KEYS)
        res["total"] = t_total
        res["iterations"] = self._iterations
        res["num_prompts"] = self.num_prompts

        print(f"  total: {t_total}s, {self._iterations} iterations, {self.num_prompts} prompt(s)")
        print()
        return res

@dataclasses.dataclass
class Plan:
    action_plan: Optional[List[ActionCall]]
    objects: Optional[Union[Mapping[str, Object], StreamConstraint]]
    log: Log


class Algorithm(abc.ABC):
    def __init__(self, max_level: int):
        self.max_level = max_level

    @abc.abstractmethod
    def plan(
        self,
        pddl: symbolic.Pddl,
        streams: Sequence[Stream],
        actions: Sequence[Action],
        geometric_predicates: Sequence[_P],
        stream_state: Set[str],
        objects: Sequence[Object],
        world: Any,
        constraint_cls: Type[Constraint],
    ) -> Plan:
        pass
