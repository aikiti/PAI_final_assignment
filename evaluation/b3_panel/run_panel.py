#!/usr/bin/env python3
"""B3/T3 評価ランナー: ③ 手元の配布 kit 準拠（主評価）と ① init ごとの再生成（感度分析）。Codex 2026-09-26 裁定 (a)。

判定を公式と変えないため、episode ループ・静定 10 step・成功判定・衝突判定は配布 kit の
`pipeline.rollout.RolloutExecutor.evaluate_task` をそのまま呼ぶ（複製も再実装もしない）。記録は外側の 2 つの覗き窓
（`ProbeEnv`・`ProbePolicy`）だけで取る。作りは `work/tools/qself/record_selfroll.py` と同じ。

  ③ kit   … manifest の task ごとに evaluate_task を 1 回。env を 1 個作り env.seed(env_seed)（配布 kit の create_env）、
             init 配列を manifest の行順に組んで渡し、同じ env で連続 reset。方策の reset seed は配布 kit どおり env_seed + episode_id。
  ① fresh … manifest の行ごとに evaluate_task を 1 回（init 1 行・n_eval_episodes=1）。env は毎回作り直して env.seed(env_seed)。
             方策へ渡す reset seed は ③ と同じ env_seed + episode_id に揃える（配布 kit が渡す値 env_seed + 0 も記録する）。

記録（**乱数を消費しない**。`reset_contract.py`）:
  reset ごと … reset 前後の乱数状態の digest、内側 reset の試行回数（配置失敗の再試行）、env 内の reset 通番、sim の作り直しの有無、
               家具（bddl の fixtures の root body）の位置・姿勢と digest。
  episode ごと … 最初の step の直前の状態（= 復元した init）の指紋と manifest の init との一致、静定後（方策 reset の直前）の状態の
               指紋と乱数状態、各 step の raw done、非対象物の L1 変位（公式と同じ選び方）の最大値と初回超過 step、終端の分類、
               episode 後の乱数状態。
出力（1 ファイル = 1 試行。既存ファイルには書かない。行を飛ばして再開しない）:
  run_meta → episode × N（task_end を task ごと）→ run_end。途中で止まった task は、task 単位で新しい試行（新しいファイル）で
  走らせ直し、旧ファイルは残す。

方策:
  zero      … 全 step ゼロ行動（G-ZERO。マスクの凍結に使う）
  random    … 一様乱数行動（自分専用の乱数。グローバル乱数は使わない）。行動の履歴が次の reset に響くかを見る
  rng_probe … ゼロ行動を返すが、毎 step グローバルの np.random を 1 回消費する（同じプロセスで推論したときの影響を見る陰性対照）
  server    … ローカルで起動した提出物の policy_server.py（run_selfeval と同じ ServerPolicy。content_hash を照合）

使い方（`source PARC2026_pre-main/env.sh` の後。AC 電源で）:
  python work/eval/b3_panel/run_panel.py --manifest work/reports/honsen2_b3_eval_20260926/B3_EVAL_MANIFEST_v1.json \
      --manifest-sha <凍結 sha> --mode kit --env-seed 42 --policy zero --label gzero_b3 --out <新しい jsonl>
公式スコアは計算しない。LB を予測しない。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "PARC2026_pre-main", REPO / "PARC2026_pre-main/LIBERO-plus", REPO / "PARC2026_final",
           REPO / "work/eval/selfeval", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("LIBERO_ROOT", str(REPO / "PARC2026_pre-main/LIBERO-plus"))
os.environ.setdefault("MAGICK_HOME", "/opt/homebrew")
os.environ.setdefault("MUJOCO_GL", "cgl")

import reset_contract as rc  # noqa: E402

logger = logging.getLogger("b3_panel")
SETTLE_EXPECTED = 10   # rollout.py の静定 step 数（記録側では数えるだけで、ループには関与しない）


class PanelError(RuntimeError):
    """前提が崩れたとき（黙って壊れた記録を積まないため）。"""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- 方策

class RngProbePolicy:
    """ゼロ行動を返すが、毎 step グローバルの np.random を消費する（陰性対照）。"""

    name = "rng_probe"

    def __init__(self, action_dim: int = 7) -> None:
        self._dim = action_dim

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        del instruction, seed

    def get_action(self, obs: dict) -> np.ndarray:
        np.random.random()   # ★ 意図的な消費
        return np.zeros(self._dim)


def make_policy(name: str) -> Any:
    import run_selfeval as rs
    if name == "rng_probe":
        return RngProbePolicy()
    return rs.POLICIES[name]()


# ----------------------------------------------------------------------------- 覗き窓

class Tape:
    """1 task（③）または 1 episode（①）ぶんの記録。ProbeEnv と ProbePolicy の両方から書く。"""

    def __init__(self, obj_of_interest: set[str], threshold_m: float) -> None:
        self.ooi = set(obj_of_interest)
        self.threshold = float(threshold_m)
        self.episodes: list[dict[str, Any]] = []
        self.cur: dict[str, Any] | None = None
        self.resets_in_env = 0
        self.last_sim_id: int | None = None
        self.last_obs: dict[str, Any] | None = None

    # reset（ProbeEnv から）
    def reset_begin(self) -> dict[str, Any]:
        return {"rng_pre": rc.rng_digest()}

    def reset_end(self, env: Any, begin: dict[str, Any], attempts: int) -> None:
        inner = getattr(env, "env", env)
        sim_id = id(inner.sim)
        self.resets_in_env += 1
        poses = rc.fixture_poses(env)
        self.cur = {
            "reset": {**begin, "rng_post": rc.rng_digest(), "attempts": int(attempts), "ordinal_in_env": self.resets_in_env,
                      "sim_rebuilt": (self.last_sim_id is not None and sim_id != self.last_sim_id),
                      "fixtures": poses, "fixtures_digest": rc.fixtures_digest(poses)},
            "init": None, "settle": None, "steps": 0, "settle_steps": 0, "active": False, "dones": [],
            "origin": {}, "max_disp": {}, "first_cross": {}, "act_ms": [], "policy_seed_executor": None, "policy_seed_passed": None,
        }
        self.last_sim_id = sim_id

    # step（ProbeEnv から）
    def before_step(self, env: Any) -> None:
        cur = self.cur
        if cur is None:
            raise PanelError("reset の前に step が呼ばれた")
        if cur["init"] is None:   # 最初の step の直前 = set_state_from_flattened＋forward の直後
            inner = getattr(env, "env", env)
            st = np.asarray(inner.sim.get_state().flatten(), dtype=np.float64)
            cur["init"] = {"observed_sha256": rc.array_sha256(st), "state": rc.state_fingerprint(inner.sim), "_vec": st}

    def after_step(self, obs: dict[str, Any], done: bool) -> None:
        cur = self.cur
        self.last_obs = obs
        if not cur["active"]:
            cur["settle_steps"] += 1
            return
        cur["steps"] += 1
        cur["dones"].append(bool(done))
        for name, p0 in cur["origin"].items():
            now = obs.get(name + "_pos")
            if now is None:
                continue
            d = float(np.sum(np.abs(np.asarray(now) - p0)))     # rollout.py と同じ L1
            if d > cur["max_disp"][name]:
                cur["max_disp"][name] = d
            if cur["first_cross"][name] is None and d > self.threshold:
                cur["first_cross"][name] = cur["steps"]

    # policy.reset（ProbePolicy から）= 静定が終わった時点
    def policy_reset(self, env_sim: Any, executor_seed: int | None, passed_seed: int | None) -> None:
        cur = self.cur
        if cur is None or self.last_obs is None:
            raise PanelError("policy.reset の前に reset・step が無い")
        obs = self.last_obs
        cur["origin"] = {k[:-4]: np.asarray(obs[k]).copy() for k in obs
                         if k.endswith("_pos") and not k.startswith("robot0") and not k.endswith("_to_robot0_eef_pos")
                         and k[:-4] not in self.ooi}                       # rollout.py と同じ選び方
        cur["max_disp"] = {n: 0.0 for n in cur["origin"]}
        cur["first_cross"] = {n: None for n in cur["origin"]}
        cur["settle"] = {"steps": int(cur["settle_steps"]), "state": rc.state_fingerprint(env_sim), "rng": rc.rng_digest()["np"]}
        cur["policy_seed_executor"], cur["policy_seed_passed"] = executor_seed, passed_seed
        cur["active"] = True

    def episode_end(self) -> None:
        if self.cur is not None:
            self.cur["rng_after_episode"] = rc.rng_digest()
            self.episodes.append(self.cur)
            self.cur = None


class ProbeEnv:
    """env の外側。reset と step の前後に指紋を取り、他は素通し。"""

    def __init__(self, env: Any, tape: Tape) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_tape", tape)
        inner = getattr(env, "env", env)
        counter = {"n": 0}
        original = inner.reset

        def counting_reset(*args: Any, **kwargs: Any) -> Any:   # ControlEnv.reset は RandomizationError の間これを繰り返す
            counter["n"] += 1
            return original(*args, **kwargs)

        inner.reset = counting_reset                             # インスタンスにだけ被せる（クラス・配布物は触らない）
        object.__setattr__(self, "_counter", counter)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_env"), name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        tape: Tape = self._tape
        if tape.cur is not None:
            tape.episode_end()
        begin = tape.reset_begin()
        n0 = self._counter["n"]
        obs = self._env.reset(*args, **kwargs)
        tape.reset_end(self._env, begin, self._counter["n"] - n0)
        return obs

    def step(self, action: Any) -> Any:
        tape: Tape = self._tape
        tape.before_step(self._env)
        result = self._env.step(action)
        obs, _reward, done, _info = result
        tape.after_step(obs, bool(done))
        return result

    def close(self) -> Any:
        tape: Tape = self._tape
        if tape.cur is not None:
            tape.episode_end()
        return self._env.close()


class ProbePolicy:
    """方策の外側。reset で静定後の指紋を取る。① では方策へ渡す seed を ③ と同じ値に揃える。"""

    def __init__(self, inner: Any, tape: Tape, env_holder: dict, seed_for_episode: list[int]) -> None:
        self._inner, self._tape, self._env_holder = inner, tape, env_holder
        self._seeds = list(seed_for_episode)
        self._k = 0
        self.name = getattr(inner, "name", "unknown")

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        passed = self._seeds[self._k] if self._k < len(self._seeds) else seed
        self._k += 1
        env = self._env_holder["env"]
        self._tape.policy_reset(getattr(env, "env", env).sim, seed, passed)
        self._tape.cur["instruction"] = instruction
        self._inner.reset(instruction=instruction, seed=passed)

    def get_action(self, obs: dict) -> np.ndarray:
        t0 = time.perf_counter()
        a = self._inner.get_action(obs)
        self._tape.cur["act_ms"].append((time.perf_counter() - t0) * 1000.0)
        return a


# ----------------------------------------------------------------------------- 実行

def runtime_info(threshold: float, max_steps: int, camera: int) -> dict[str, Any]:
    import mujoco
    import robosuite
    import pipeline.environment as env_mod
    import pipeline.rollout as rollout_mod
    import run_selfeval as rs
    return {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__,
            "robosuite": getattr(robosuite, "__version__", "?"),
            "libero_plus_rev": rs._git_rev(REPO / "PARC2026_pre-main/LIBERO-plus"),
            "rollout_path": str(Path(rollout_mod.__file__).resolve().relative_to(REPO)), "rollout_sha256": sha256_file(Path(rollout_mod.__file__)),
            "environment_path": str(Path(env_mod.__file__).resolve().relative_to(REPO)), "environment_sha256": sha256_file(Path(env_mod.__file__)),
            "collision_threshold_m": threshold, "max_steps": max_steps, "camera": camera,
            "control_freq": int(os.environ.get("LIBERO_CONTROL_FREQ", "20")), "platform": sys.platform, "machine": platform.machine()}


def make_task_info(task: dict[str, Any], init_rows: np.ndarray) -> Any:
    from pipeline.environment import TaskInfo
    return TaskInfo(task_id=int(task["task_order"]), name=task["task_stem"], language=task["language"],
                    bddl_file=str(REPO / task["bddl"]), init_states=init_rows, benchmark_name="b3_panel_manifest")


def episode_record(meta: dict[str, Any], task: dict[str, Any], ep_spec: dict[str, Any], tape_ep: dict[str, Any],
                   result: Any, init_vec: np.ndarray, executor_episode_id: int) -> dict[str, Any]:
    max_steps = meta["runtime"]["max_steps"]
    done = bool(tape_ep["dones"][-1]) if tape_ep["dones"] else False
    collided_re = any(v > meta["runtime"]["collision_threshold_m"] for v in tape_ep["max_disp"].values())
    term = rc.classify_termination(done, bool(result.collided), int(result.total_steps), max_steps)
    observed = tape_ep["init"]
    init_match = bool(observed is not None and observed["_vec"].shape == init_vec.shape and np.array_equal(observed["_vec"], init_vec))
    first = {n: s for n, s in tape_ep["first_cross"].items() if s is not None}
    env_seed = meta["env_seed"]
    want_exec_seed = env_seed + (int(ep_spec["episode_id"]) if meta["mode"] == "kit" else 0)
    crosscheck = {
        "steps_equal_official": int(tape_ep["steps"]) == int(result.total_steps),
        "collided_equals_recomputed": bool(collided_re) == bool(result.collided),
        "success_equals_done_and_not_collided": bool(result.success) == (done and not bool(result.collided)),
        "settle_steps_expected": int(tape_ep["settle_steps"]) == SETTLE_EXPECTED,
        "init_state_equals_manifest": init_match,
        "init_sha_equals_manifest": bool(observed is not None and observed["observed_sha256"] == ep_spec["init_state_sha256"]),
        "executor_seed_expected": tape_ep["policy_seed_executor"] == want_exec_seed,
        "passed_seed_is_env_seed_plus_episode_id": tape_ep["policy_seed_passed"] == env_seed + int(ep_spec["episode_id"]),
        "instruction_equals_manifest": tape_ep.get("instruction") == task["language"],
    }
    rec = {
        "kind": "episode", "trial_id": meta["trial_id"], "label": meta["label"], "policy": meta["policy"],
        "manifest_sha256": meta["manifest"]["sha256"], "manifest_name": meta["manifest"]["name"],
        "reset_mode": meta["mode"], "env_seed": env_seed,
        "task_order": int(task["task_order"]), "task_id": task["task_id"], "family": task.get("family"), "group": task.get("group"),
        "task_stem": task["task_stem"], "bddl_sha256": task["bddl_sha256"], "init_file_sha256": task["init_file_sha256"],
        "episode_id": int(ep_spec["episode_id"]), "init_index": int(ep_spec["init_index"]), "init_state_sha256": ep_spec["init_state_sha256"],
        "executor_episode_id": int(executor_episode_id), "reset_ordinal_in_env": int(tape_ep["reset"]["ordinal_in_env"]),
        "policy_reset_seed_executor": tape_ep["policy_seed_executor"], "policy_reset_seed_passed": tape_ep["policy_seed_passed"],
        "reset": tape_ep["reset"],
        "init": {k: v for k, v in (observed or {}).items() if k != "_vec"},
        "settle": tape_ep["settle"],
        "done_raw": done, "collided": bool(result.collided), "success": bool(result.success),
        "safe_success": bool(result.success) and not bool(result.collided),
        "steps": int(result.total_steps), "max_steps": max_steps, "termination": term,
        "nontarget": {"monitored": sorted(tape_ep["origin"]), "max_disp_m": tape_ep["max_disp"], "first_cross_step": first,
                      "first_violation_step": (min(first.values()) if first else None), "collided_recomputed": bool(collided_re)},
        "rng_after_episode": tape_ep.get("rng_after_episode"),
        "elapsed_sec": round(float(result.elapsed_time_sec), 3),
        "act_ms_p50": (float(np.median(tape_ep["act_ms"])) if tape_ep["act_ms"] else None),
        "act_ms_max": (float(np.max(tape_ep["act_ms"])) if tape_ep["act_ms"] else None),
        "crosscheck": crosscheck, "crosscheck_ok": all(crosscheck.values()),
    }
    return rec


def run(a: argparse.Namespace) -> int:
    import run_selfeval as rs
    from compe.t3 import register_t3
    from libero.libero.envs.object_states.base_object_states import SiteObjectState
    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.rollout import RolloutExecutor

    manifest_path = Path(a.manifest).resolve()
    manifest_sha = sha256_file(manifest_path)
    if a.manifest_sha and manifest_sha != a.manifest_sha:
        raise PanelError(f"manifest の sha が指定値と違う: {manifest_sha}")
    manifest = json.loads(manifest_path.read_text())
    out = Path(a.out)
    if out.exists():
        raise PanelError(f"出力が既にある（追記・再開はしない。新しいファイル名で走らせる）: {out}")
    if a.require_ac and not rs_on_ac():
        raise PanelError("AC 電源でない（バッテリーでは step が数十倍遅くなる。2026-09-09 の実害）")

    env_meta = rs.check_environment()
    guard_t1 = rs.install_support_contact_guard()
    register_t3()
    guards = {"support_contact_guard(t1)": bool(guard_t1),
              "table_contact_guard(t3)": bool(getattr(SiteObjectState, "_t3_table_contact_guard", False))}
    if not all(guards.values()):
        raise PanelError(f"接触ガードが揃っていない: {guards}")

    if a.policy == "server":
        if a.policy_server_artifact is None or a.policy_server_log is None:
            raise PanelError("--policy server には --policy-server-artifact と --policy-server-log が要る")
        rs.ServerPolicy.configure(a.policy_server_url, a.policy_server_artifact, timeout_sec=a.policy_server_timeout,
                                  server_log=a.policy_server_log)

    eval_config = EvalConfig()
    eval_config.max_steps_per_episode = a.max_steps
    eval_config.seed = a.env_seed
    eval_config.camera_height = eval_config.camera_width = 128
    env_manager = EnvironmentManager(eval_config)
    executor = RolloutExecutor(env_manager, eval_config)
    perturbation = PerturbationConfig()
    if perturbation.observation_noise > 0 or perturbation.action_noise > 0:
        raise PanelError("PerturbationConfig の既定にノイズが入っている（前提と違う）")
    cc = executor.scoring_config.get("collision", {})
    if not bool(cc.get("enabled", True)):
        raise PanelError("collision.enabled が false")
    threshold = float(cc.get("threshold_m", 0.001))

    tasks = manifest["tasks"]
    if a.tasks:
        want = set(a.tasks)
        unknown = want - {t["task_id"] for t in tasks}
        if unknown:
            raise PanelError(f"manifest に無い task: {sorted(unknown)}")
        tasks = [t for t in tasks if t["task_id"] in want]
    if a.reverse_tasks:
        tasks = list(reversed(tasks))

    trial_id = f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}_{os.urandom(3).hex()}"   # os.urandom は乱数状態を触らない
    meta = {
        "kind": "run_meta", "schema": 1, "trial_id": trial_id, "label": a.label, "mode": a.mode, "env_seed": a.env_seed,
        "policy": a.policy, "reverse_tasks": bool(a.reverse_tasks), "reverse_within_task": bool(a.reverse_within_task),
        "task_filter": a.tasks,
        "policy_descriptor": (f"server:{rs.ServerPolicy.url}:content_hash={rs.ServerPolicy.artifact_content_hash}"
                              f":server_hash_verified={rs.ServerPolicy.server_hash_verified}" if a.policy == "server" else f"builtin:{a.policy}"),
        "server_log": str(a.policy_server_log) if a.policy_server_log else None,
        "manifest": {"path": str(manifest_path.relative_to(REPO)), "sha256": manifest_sha, "name": manifest["name"]},
        "contract": manifest["contract"].get(a.mode),
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime": runtime_info(threshold, a.max_steps, 128), "guards": guards, "env": env_meta,
        "perturbation": {"observation_noise": perturbation.observation_noise, "action_noise": perturbation.action_noise},
        "code": {"run_panel_sha256": sha256_file(Path(__file__)), "reset_contract_sha256": sha256_file(HERE / "reset_contract.py"),
                 "run_selfeval_sha256": sha256_file(REPO / "work/eval/selfeval/run_selfeval.py")},
        "repo_rev": rs._git_rev(REPO),
    }
    meta["runtime_fingerprint"] = rc.runtime_fingerprint(meta)
    # 判定に効くコードの来歴（Codex 2026-09-26 P2）: 配布 kit の判定・接触ガード・採点の既定値のファイルと、LIBERO-plus の
    # 作業ツリー（改変を含む）。G-ZERO マスクの来歴（sidecar）と照合し、違えば昇格判定に使わない（tables.py）。
    # ★ 名前は runtime_contract（"contract" は manifest の reset 方式の説明文で、別物。2026-09-26 に取り違えかけた）
    meta["runtime_contract"] = rc.runtime_contract(REPO)
    meta["contract_fingerprint"] = rc.contract_fingerprint(meta["runtime_contract"])
    if a.binding:
        bind = json.loads(Path(a.binding).read_text())
        meta["binding"] = {"path": str(Path(a.binding).resolve().relative_to(REPO)), "sha256": sha256_file(Path(a.binding)), **bind}
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = n_bad = 0
    started = time.time()
    with out.open("x", encoding="utf-8") as sink:
        def emit(obj: dict[str, Any]) -> None:
            sink.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sink.flush()

        emit(meta)
        for task in tasks:
            eps = list(task["episodes"])
            if a.reverse_within_task:
                if a.mode == "kit":
                    raise PanelError("③ で task 内の順を入れ替えると凍結した契約から外れる（① の順序確認だけで使う）")
                eps = list(reversed(eps))
            import torch
            init_all = np.asarray(torch.load(str(REPO / task["init_file"]), weights_only=False), dtype=np.float64)
            if sha256_file(REPO / task["init_file"]) != task["init_file_sha256"] or sha256_file(REPO / task["bddl"]) != task["bddl_sha256"]:
                raise PanelError(f"{task['task_id']}: bddl か init ファイルの sha が manifest と違う")
            groups = [eps] if a.mode == "kit" else [[e] for e in eps]
            t_task = time.time()
            task_status, task_err = "ok", None
            for grp in groups:
                rows = np.stack([init_all[int(e["init_index"])] for e in grp])
                info = make_task_info(task, rows)
                eval_config.n_eval_episodes = len(grp)
                ooi = set(env_manager.get_obj_of_interest(info))
                tape = Tape(ooi, threshold)
                holder: dict[str, Any] = {}
                original_create = env_manager.create_env

                def create_probed(ti: Any, _orig=original_create, _tape=tape, _holder=holder) -> ProbeEnv:
                    env = _orig(ti)                       # 配布 kit の create_env（OffScreenRenderEnv ＋ env.seed(config.seed)）
                    pe = ProbeEnv(env, _tape)
                    _holder["env"] = env
                    return pe

                env_manager.create_env = create_probed    # インスタンスにだけ被せる
                policy = ProbePolicy(make_policy(a.policy), tape, holder,
                                     [a.env_seed + int(e["episode_id"]) for e in grp])
                try:
                    result = executor.evaluate_task(policy, info, perturbation)
                except Exception as exc:  # noqa: BLE001 — task 単位で記録して次へ（途中の task は再開しない）
                    task_status, task_err = "error", f"{type(exc).__name__}: {str(exc)[:400]}"
                    emit({"kind": "task_error", "trial_id": trial_id, "task_id": task["task_id"], "error": task_err,
                          "traceback": traceback.format_exc()[-1500:], "episodes_completed_in_group": len(tape.episodes)})
                    n_err += 1
                    break
                finally:
                    env_manager.create_env = original_create
                if len(result.episodes) != len(grp) or len(tape.episodes) != len(grp):
                    raise PanelError(f"{task['task_id']}: 記録数が合わない（公式 {len(result.episodes)}・記録 {len(tape.episodes)}・予定 {len(grp)}）")
                for k, (e, res, tep) in enumerate(zip(grp, result.episodes, tape.episodes)):
                    rec = episode_record(meta, task, e, tep, res, init_all[int(e["init_index"])], res.episode_id)
                    emit(rec)
                    n_ok += 1
                    n_bad += (not rec["crosscheck_ok"])
                    print(f"[b3] {task['task_id']} ep{rec['episode_id']} init{rec['init_index']} {rec['termination']} "
                          f"T={rec['steps']} coll={rec['collided']} fx={rec['reset']['fixtures_digest']} "
                          f"try={rec['reset']['attempts']} ord={rec['reset_ordinal_in_env']} ok={rec['crosscheck_ok']}", flush=True)
            emit({"kind": "task_end", "trial_id": trial_id, "task_id": task["task_id"], "status": task_status, "error": task_err,
                  "wall_sec": round(time.time() - t_task, 1)})
            if a.require_ac and not rs_on_ac():
                emit({"kind": "run_end", "trial_id": trial_id, "status": "stopped_not_on_ac", "episodes_ok": n_ok,
                      "task_errors": n_err, "crosscheck_failed": n_bad, "wall_sec": round(time.time() - started, 1)})
                print("[b3] AC 電源でなくなったので止めた（次の task は始めない）")
                return 5
        emit({"kind": "run_end", "trial_id": trial_id, "status": "complete" if n_err == 0 else "task_errors",
              "episodes_ok": n_ok, "task_errors": n_err, "crosscheck_failed": n_bad, "wall_sec": round(time.time() - started, 1)})
    print(f"[b3] 完了 trial={trial_id} episodes={n_ok} task_errors={n_err} crosscheck_failed={n_bad} "
          f"{time.time() - started:.0f}s -> {out}")
    return 0 if (n_err == 0 and n_bad == 0) else 1


def rs_on_ac() -> bool:
    import subprocess
    try:
        o = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "AC Power" in o if o else True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-sha", default=None, help="凍結 sha（違えば走らせない）")
    ap.add_argument("--mode", choices=("kit", "fresh"), required=True, help="kit = ③ 主評価 / fresh = ① 感度分析")
    ap.add_argument("--env-seed", type=int, default=42)
    ap.add_argument("--policy", choices=("zero", "random", "rng_probe", "server"), default="zero")
    ap.add_argument("--label", required=True, help="走行の名前（例: gzero_b3_kit_s42）")
    ap.add_argument("--out", type=Path, required=True, help="新しい jsonl（既にあれば止まる）")
    ap.add_argument("--tasks", nargs="+", default=None, help="manifest の task_id で絞る（再走は task 単位）")
    ap.add_argument("--reverse-tasks", action="store_true", help="task の順を逆にする（③ は task ごとに env を作り直すので順の影響を見る）")
    ap.add_argument("--reverse-within-task", action="store_true", help="① だけ: task 内の init の順を逆にする")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--require-ac", action="store_true")
    ap.add_argument("--policy-server-url", default="http://127.0.0.1:8000")
    ap.add_argument("--policy-server-artifact", type=Path, default=None)
    ap.add_argument("--policy-server-log", type=Path, default=None)
    ap.add_argument("--policy-server-timeout", type=float, default=10.0)
    ap.add_argument("--binding", type=Path, default=None,
                    help="label と export の結び付け（run_b3_panel.sh が作る JSON）。run_meta に写す")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("pipeline.rollout", "pipeline.environment"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        return run(a)
    except PanelError as exc:
        print(f"[b3] ★ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
