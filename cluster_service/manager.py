from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import ServiceConfig
from .experiments import (
    build_judgment_template,
    evaluate_with_judgments,
    run_experiment,
    select_example_queries,
)
from .pipeline import BuildArtifacts, load_build, run_build
from .search_adapter import SearchAdapterConfig
from .utils import read_json, utc_now_iso, write_json


class ServiceManager:
    def __init__(self, cfg: ServiceConfig | None = None) -> None:
        self.cfg = cfg or ServiceConfig()
        self.cfg.ensure_directories()
        self._lock = threading.Lock()
        self._build_executor = ThreadPoolExecutor(
            max_workers=self.cfg.build_executor_workers,
            thread_name_prefix="cluster-build",
        )
        self._experiment_executor = ThreadPoolExecutor(
            max_workers=self.cfg.experiment_executor_workers,
            thread_name_prefix="cluster-exp",
        )
        self._state = self._load_state()
        self._build_cache: dict[str, BuildArtifacts] = {}
        self._experiment_cache: dict[str, dict[str, Any]] = {}
        self._startup_status: dict[str, Any] = {"status": "not_started"}

    def _default_state(self) -> dict[str, Any]:
        return {"current_build_id": None, "builds": {}, "experiments": {}}

    def _load_state(self) -> dict[str, Any]:
        if not self.cfg.state_path.exists():
            return self._default_state()
        payload = read_json(self.cfg.state_path)
        if not isinstance(payload, dict):
            return self._default_state()
        state = self._default_state()
        state.update(payload)
        state.setdefault("builds", {})
        state.setdefault("experiments", {})
        return state

    def _save_state(self) -> None:
        write_json(self.cfg.state_path, self._state)

    def _update_entry(self, bucket: str, entry_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            entry = deepcopy(self._state[bucket].get(entry_id, {"id": entry_id}))
            entry.update(updates)
            entry["updated_at"] = utc_now_iso()
            self._state[bucket][entry_id] = entry
            self._save_state()
            return deepcopy(entry)

    def _build_dir(self, build_id: str) -> Path:
        return self.cfg.output_root / "builds" / build_id

    def _experiment_dir(self, run_id: str) -> Path:
        return self.cfg.output_root / "experiments" / run_id

    def start_build(self, search_adapter_payload: dict[str, Any] | None, make_current: bool) -> dict[str, Any]:
        build_id = uuid4().hex[:12]
        entry = self._update_entry(
            "builds",
            build_id,
            id=build_id,
            status="queued",
            created_at=utc_now_iso(),
            make_current=make_current,
        )
        self._build_executor.submit(
            self._run_build_job,
            build_id,
            search_adapter_payload or {},
            make_current,
        )
        return entry

    def _run_build_job(
        self, build_id: str, search_adapter_payload: dict[str, Any], make_current: bool
    ) -> None:
        self._update_entry("builds", build_id, status="running")
        try:
            manifest = run_build(build_id, self.cfg, search_adapter_payload)
            if make_current:
                with self._lock:
                    self._state["current_build_id"] = build_id
                    self._save_state()
            self._update_entry(
                "builds",
                build_id,
                status="completed",
                build_id=build_id,
                manifest=manifest,
                build_dir=str(self._build_dir(build_id)),
            )
        except Exception as exc:
            self._update_entry("builds", build_id, status="failed", error=str(exc))

    def get_build_status(self, build_id: str) -> dict[str, Any]:
        entry = deepcopy(self._state["builds"].get(build_id))
        if entry:
            return entry
        manifest_path = self._build_dir(build_id) / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            return {
                "id": build_id,
                "status": manifest.get("status", "completed"),
                "manifest": manifest,
                "build_dir": str(self._build_dir(build_id)),
            }
        raise KeyError(f"Unknown build id: {build_id}")

    def current_build_id(self) -> str | None:
        return self._state.get("current_build_id")

    def startup_status(self) -> dict[str, Any]:
        return deepcopy(self._startup_status)

    def _set_current_build(self, build_id: str) -> None:
        with self._lock:
            self._state["current_build_id"] = build_id
            self._save_state()

    def ensure_startup_build(self) -> dict[str, Any]:
        self._startup_status = {"status": "running", "started_at": utc_now_iso()}

        current_id = self.current_build_id()
        if current_id:
            build_dir = self._build_dir(current_id)
            manifest_path = build_dir / "manifest.json"
            if manifest_path.exists():
                self.load_build(current_id)
                self._startup_status = {
                    "status": "loaded_existing",
                    "build_id": current_id,
                    "updated_at": utc_now_iso(),
                }
                return deepcopy(self._startup_status)

        completed_builds = sorted(
            (
                path
                for path in (self.cfg.output_root / "builds").iterdir()
                if path.is_dir() and (path / "manifest.json").exists()
            ),
            key=lambda path: (path / "manifest.json").stat().st_mtime,
            reverse=True,
        )
        if completed_builds:
            build_id = completed_builds[0].name
            self._set_current_build(build_id)
            self.load_build(build_id)
            self._startup_status = {
                "status": "loaded_existing",
                "build_id": build_id,
                "updated_at": utc_now_iso(),
            }
            return deepcopy(self._startup_status)

        build_id = "startup-build"
        self._update_entry(
            "builds",
            build_id,
            id=build_id,
            status="running",
            created_at=utc_now_iso(),
            make_current=True,
        )
        try:
            manifest = run_build(
                build_id,
                self.cfg,
                {"url": self.cfg.default_search_api_url},
            )
            self._set_current_build(build_id)
            self._update_entry(
                "builds",
                build_id,
                status="completed",
                build_id=build_id,
                manifest=manifest,
                build_dir=str(self._build_dir(build_id)),
            )
            self.load_build(build_id)
            self._startup_status = {
                "status": "built_on_startup",
                "build_id": build_id,
                "updated_at": utc_now_iso(),
            }
            return deepcopy(self._startup_status)
        except Exception as exc:
            self._update_entry("builds", build_id, status="failed", error=str(exc))
            self._startup_status = {
                "status": "failed",
                "build_id": build_id,
                "error": str(exc),
                "updated_at": utc_now_iso(),
            }
            raise

    def load_build(self, build_id: str | None = None) -> BuildArtifacts:
        resolved = build_id or self.current_build_id()
        if not resolved:
            raise RuntimeError("No completed build is available.")
        cached = self._build_cache.get(resolved)
        if cached is not None:
            return cached
        build_dir = self._build_dir(resolved)
        if not build_dir.exists():
            raise FileNotFoundError(f"Build directory not found: {build_dir}")
        artifacts = load_build(build_dir)
        self._build_cache[resolved] = artifacts
        return artifacts

    def start_experiment(
        self,
        build_id: str | None,
        baseline_method: str,
        top_k: int,
        search_adapter_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run_id = uuid4().hex[:12]
        entry = self._update_entry(
            "experiments",
            run_id,
            id=run_id,
            status="queued",
            created_at=utc_now_iso(),
            build_id=build_id or self.current_build_id(),
        )
        self._experiment_executor.submit(
            self._run_experiment_job,
            run_id,
            build_id,
            baseline_method,
            top_k,
            search_adapter_payload or {},
        )
        return entry

    def _run_experiment_job(
        self,
        run_id: str,
        build_id: str | None,
        baseline_method: str,
        top_k: int,
        search_adapter_payload: dict[str, Any],
    ) -> None:
        self._update_entry("experiments", run_id, status="running")
        try:
            build = self.load_build(build_id)
            adapter = SearchAdapterConfig.from_payload(search_adapter_payload, self.cfg)
            payload = run_experiment(
                run_id=run_id,
                build=build,
                benchmark_path=self.cfg.benchmark_path,
                adapter=adapter,
                baseline_method=baseline_method,
                top_k=top_k,
                output_dir=self._experiment_dir(run_id),
            )
            self._experiment_cache[run_id] = payload
            self._update_entry(
                "experiments",
                run_id,
                status="completed",
                build_id=build.build_id,
                summary=payload["summary"],
                run_dir=str(self._experiment_dir(run_id)),
            )
        except Exception as exc:
            self._update_entry("experiments", run_id, status="failed", error=str(exc))

    def get_experiment(self, run_id: str) -> dict[str, Any]:
        cached = self._experiment_cache.get(run_id)
        if cached is not None:
            return cached
        run_dir = self._experiment_dir(run_id)
        summary_path = run_dir / "summary.json"
        per_query_path = run_dir / "per_query.json"
        if summary_path.exists() and per_query_path.exists():
            payload = {
                "summary": read_json(summary_path),
                "per_query": read_json(per_query_path),
            }
            judged_path = run_dir / "judged_metrics.json"
            if judged_path.exists():
                payload["judged"] = read_json(judged_path)
            self._experiment_cache[run_id] = payload
            return payload
        entry = self._state["experiments"].get(run_id)
        if entry:
            return {"status": entry.get("status"), "detail": entry}
        raise KeyError(f"Unknown experiment id: {run_id}")

    def build_judgment_template(self, run_id: str) -> dict[str, Any]:
        payload = self.get_experiment(run_id)
        if "per_query" not in payload:
            raise RuntimeError("Experiment has not completed yet.")
        template = build_judgment_template(payload, self._experiment_dir(run_id))
        self._update_entry(
            "experiments",
            run_id,
            judgment_template=template,
        )
        return template

    def evaluate_experiment(self, run_id: str, judgments: list[dict[str, Any]]) -> dict[str, Any]:
        payload = self.get_experiment(run_id)
        if "per_query" not in payload:
            raise RuntimeError("Experiment has not completed yet.")
        judged = evaluate_with_judgments(payload, judgments, self._experiment_dir(run_id))
        payload["judged"] = judged
        payload["summary"]["judged_metrics"] = judged["summary"]
        payload["summary"]["example_queries"] = select_example_queries(
            payload["per_query"],
            use_judged=True,
            judged_payload=judged,
        )
        write_json(self._experiment_dir(run_id) / "summary.json", payload["summary"])
        self._experiment_cache[run_id] = payload
        self._update_entry(
            "experiments",
            run_id,
            judged_summary=judged["summary"],
        )
        return judged
