# VIZ-COAST

Code for **Using VLM Reasoning to Constrain Task and Motion Planning**
([arXiv:2510.25548](https://arxiv.org/abs/2510.25548), [PDF](https://arxiv.org/pdf/2510.25548)).

**Authors:** Muyang Yan, Miras Mengdibayev, Ardon Floros, Weihang Guo, Lydia E. Kavraki, Zachary Kingston

VIZ-COAST uses a vision-language model to predict downward-refinement failures
*before* planning and encodes them as task-planning constraints. It builds on
[COAST](https://arxiv.org/abs/2405.08572) ([code](https://github.com/branvu/coast)) by Brandon Vu, Toki Migimatsu and Jeannette Bohg.

## Environment setup

### Clone with submodules
All external dependencies live in `external/` as git submodules pinned to the
exact commits used for the paper:
```
git clone --recursive https://github.com/CoMMALab/VIZ-COAST.git
cd VIZ-COAST
```
If you already cloned without `--recursive`, run `git submodule update --init --recursive`.

| submodule | upstream | pinned commit |
| --- | --- | --- |
| `external/pybullet-planning` | [branvu/pybullet-planning](https://github.com/branvu/pybullet-planning) (`coast_changes`) | `7a6a99c` |
| `external/symbolic` | [branvu/symbolic](https://github.com/branvu/symbolic) | `47ea48e` |
| `external/spatial-dyn` | [tmigimatsu/spatial-dyn](https://github.com/tmigimatsu/spatial-dyn) | `6d56d66` |
| `external/pddlboat` | [CoMMALab/pddlboat](https://github.com/CoMMALab/pddlboat) (`muyang`) | `1adef23` |
| `external/downward` | [aibasel/downward](https://github.com/aibasel/downward) | `bc6cf7a` |

`external/pddlboat` is our fork of pddlboat, which adds the Madagascar SAT
encoding used by the Z3-based planners.

### Apply the VIZ-COAST patches
VIZ-COAST depends on a few changes to the upstream libraries. These live in
`patches/` and are applied to the submodules by:
```
scripts/apply_patches.sh
```
The script is idempotent: it skips patches that are already applied.

| patch | changes |
| --- | --- |
| `pybullet-planning.patch` | Python 3.10 `collections.abc` fix, shorter grasp approach pose, larger RRT budget (`restarts=40`, `iterations=200`), robot self-collisions disabled, placement collision check removed, plus the robot/object URDFs used in the experiments (including `iiwa14_polytope_collision.urdf`) |
| `symbolic.patch` | build as C++17; `#include <cstdint>` for GCC 11+ |
| `symbolic-ctrl-utils.patch`, `spatial-dyn-ctrl-utils.patch` | `ctrl_utils/json.h`: make the Eigen `from_json` overload an exact match, so `nlohmann::json` 3.6.1 does not treat `Eigen::Matrix` as an STL container |
| `spatial-dyn.patch` | build as C++17; drop the `setuptools.extern` import that newer setuptools removed; force the pybind11 ABI tag `_cxxabi1014` to match the prebuilt `ctrlutils` wheel, so `Eigen::Quaterniond` can be shared across modules |

### Create Conda environment
```
conda create -n vizcoast python=3.10
conda activate vizcoast

conda install -c conda-forge cmake=3.25 "pybind11<3"
conda install -c conda-forge boost-cpp bison autoconf z3-solver
```

### Install the external packages
These are compiled C++ extensions, so build them from source on the target
machine. Prebuilt `.so` files or `site-packages` copied from another machine can
fail in subtle ways.
```
pip install -e external/pybullet-planning
pip install -e external/symbolic
pip install -e external/spatial-dyn
pip install -e .
```

### Build pddlboat (required; provides the Z3/Madagascar planners)
```
cd external/pddlboat
mkdir -p build/release && cd build/release
cmake ../..
make
cd ../..
pip install -e .
cd ../..
```

### Build Fast Downward (only needed for `coast_fd`)
```
cd external/downward
python build.py
cd ../..
```


## Running experiments

Set the API key for whichever provider your config uses:
```
export OPENAI_API_KEY=...    # provider: openai
export GEMINI_API_KEY=...    # provider: gemini
export QWEN_API_KEY=...      # provider: qwen
```

Run from the repo root (paths in the configs are relative to it). Each
environment has its own `run.py` and a directory of experiment configs:

```
python envs/shelf/run.py      --cfg envs/shelf/main_table/VIZ-COAST.json
python envs/cooking/run.py    --cfg envs/cooking/main_table/VIZ-COAST.json
python envs/packing/run.py    --cfg envs/packing/main_table/VIZ-COAST.json
python envs/containers/run.py --cfg envs/containers/main_table/VIZ-COAST.json
```

Swap the config to change the method. `main_table/` holds the baselines
(`VIZ-COAST`, `VIZ-COAST-nofeedback`, `coast_fd`, `coast_z3`, `incremental`,
`VLMB`); `envs/shelf/vlm_table/` holds the per-model/effort ablation, with
`MODEL_EFFORT.json` as a template to copy and fill in.

### Config format
Each config has a `global` block of shared settings and a `specific` list of
one entry per problem instance (e.g. `num_blocks`, or `goals` for cooking).
Key fields:

| field | values |
| --- | --- |
| `method` | `zoast` (internal name for VIZ-COAST), `coast_fd`, `coast_z3`, `incremental`, `VLMB` |
| `provider` | `openai`, `gemini`, `qwen` |
| `model` | provider-specific model id |
| `reasoning_effort` | `none`, `low`, `medium`, `high` |
| `prompt_strategy` | `no constraints`, `on failure`, `oracle` |
| `feedback` | `true`, `false` |
| `reps` | repetitions per instance |
| `timeout` | per-instance planning timeout (s) |
| `results_file` | output directory |

### Results
With `trial_mode: true`, results are written to
`<results_file>/<config name>/<n>.csv`, where `n` auto-increments so reruns
never overwrite a previous trial. Use `--continue-from <csv>` to resume an
interrupted run (it appends and skips completed experiments), or `--no-write`
to discard output entirely.

### Useful flags
| flag | effect |
| --- | --- |
| `--use-gui` | show the PyBullet viewer |
| `--skip-execution` | plan only, don't execute on the robot |
| `--show-thinking` | print the VLM's chain-of-thought (`VLMB` only) |
| `--reps N` / `--timeout N` | override the config values |
| `--random-seed N` | fix the seed |

Shelf adds `--auto` (skip the "Begin planning?" / "Execute?" prompts and run
straight through — recommended for batch runs), `--debug` (label grid tiles
with their PDDL names in the GUI), `--no-shelf`, and `--skip-idx N`.



## Tuning notes

### RRT motion-planning budget
If stream sampling reports motion queries as infeasible too readily (the RRT
"runs out of samples too fast"), increase the per-query budget. The two knobs
are `restarts` (number of full RRT attempts before declaring infeasible) and
`iterations` (samples per attempt), passed into the `plan_joint_motion(...)`
calls in `external/pybullet-planning/pybullet_tools/kuka_primitives.py`
(`get_free_motion_gen` and `get_holding_motion_gen`, marked `# VIZ-COAST EDITED`
by `patches/pybullet-planning.patch`). They default to the `motion_planners`
package values (`restarts=20`, `iterations=40`); the patch raises them to
`restarts=40`, `iterations=200`.
These may be tuned up further for harder/cluttered instances — at the cost of
higher worst-case planning time on genuinely infeasible configs.

## Acknowledgements
VIZ-COAST builds on the work of many others:
- [COAST](https://github.com/branvu/coast) by Brandon Vu, Toki Migimatsu and Jeannette Bohg, which VIZ-COAST's planner extends.
- [pybullet-planning](https://github.com/caelan/pybullet-planning) and [motion-planners](https://github.com/caelan/motion-planners) by Caelan Garrett, used through Brandon Vu's forks.
- [symbolic](https://github.com/tmigimatsu/symbolic), [spatial-dyn](https://github.com/tmigimatsu/spatial-dyn) and [ctrl-utils](https://github.com/tmigimatsu/ctrl-utils) by Toki Migimatsu.
- [pddlboat](https://github.com/CoMMALab/pddlboat) by Zachary Kingston, Wil Thomason, Andrew Wells and contributors; our fork adds a SAT encoding based on Jussi Rintanen's Madagascar planner.
- [Fast Downward](https://github.com/aibasel/downward) by Malte Helmert and contributors.

## Citation
```bibtex
@misc{yan2025vizcoast,
  title         = {Using {VLM} Reasoning to Constrain Task and Motion Planning},
  author        = {Yan, Muyang and Mengdibayev, Miras and Floros, Ardon and Guo, Weihang and Kavraki, Lydia E. and Kingston, Zachary},
  year          = {2025},
  eprint        = {2510.25548},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## License
MIT; see `LICENSE`. The submodules in `external/` keep their own licenses.
Fast Downward in particular is GPL-3.0; it is built and run as a separate
program and is not redistributed in this repository.
