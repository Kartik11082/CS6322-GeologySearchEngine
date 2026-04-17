from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from cluster_service.manager import ServiceManager

from .test_pipeline import _test_config, _write_fixture_corpus

app_module = importlib.import_module("cluster_service.app")


def test_rerank_endpoint_returns_ui_ready_payload(tmp_path, monkeypatch):
    pages_path, graph_path = _write_fixture_corpus(tmp_path)
    cfg = _test_config(tmp_path, pages_path, graph_path)
    manager = ServiceManager(cfg)
    manager._run_build_job("apitestbuild", {"url": "http://search.test/api"}, True)

    monkeypatch.setattr(app_module, "manager", manager)

    def fake_search_documents(query, baseline_method, top_k, adapter):
        build = manager.load_build("apitestbuild")
        rows = []
        for rank, (normalized_url, info) in enumerate(list(build.assignments.items())[:top_k], start=1):
            rows.append(
                {
                    "rank": rank,
                    "title": info["title"],
                    "url": info["url"],
                    "normalized_url": normalized_url,
                    "snippet": "",
                    "score": 1.0 / rank,
                }
            )
        return rows

    monkeypatch.setattr(app_module, "search_documents", fake_search_documents)

    client = TestClient(app_module.app)
    response = client.post(
        "/v1/rerank",
        json={
            "query": "volcano lava geology",
            "cluster_method": "flat",
            "baseline_method": "combined",
            "top_k": 5,
            "build_id": "apitestbuild",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["build_id"] == "apitestbuild"
    assert len(payload["baseline"]) == 5
    assert len(payload["reranked"]) == 5
    assert isinstance(payload["clusters"], list)


def test_cluster_catalog_endpoint_uses_selected_build(tmp_path, monkeypatch):
    pages_path, graph_path = _write_fixture_corpus(tmp_path)
    cfg = _test_config(tmp_path, pages_path, graph_path)
    manager = ServiceManager(cfg)
    manager._run_build_job("catalogbuild", {"url": "http://search.test/api"}, True)
    monkeypatch.setattr(app_module, "manager", manager)

    client = TestClient(app_module.app)
    response = client.get("/v1/clusters/flat", params={"build_id": "catalogbuild"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["build_id"] == "catalogbuild"
    assert payload["selected_k"] >= 3
    assert payload["clusters"]
