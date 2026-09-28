"""求解器与 API 的端到端测试（pytest）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    CHECK_TOL,
    ValidationError,
    solve_network,
    validate_payload,
)

client = TestClient(app)


def base_loop():
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
            {"id": "C", "type": "demand", "demand": 1.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
            {"id": "P3", "start": "B", "end": "C", "r": 1.0},
            {"id": "P4", "start": "C", "end": "S", "r": 1.0},
            {"id": "P5", "start": "S", "end": "B", "r": 2.5},
        ],
        "min_head": 10.0,
    }


def test_single_pipe_closed_form():
    """单管：q = 需水量，解析解 H_A = H_S - r·q|q|。"""
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 2.0},
        ],
        "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
        "min_head": 0.0,
    }
    res = solve_network(validate_payload(payload))
    assert res["source_supply"] == pytest.approx(2.0, abs=1e-12)
    assert res["heads"]["A"] == pytest.approx(26.0, abs=1e-9)
    assert res["flows"]["P1"] == pytest.approx(2.0, abs=1e-12)
    assert res["max_node_residual"] < CHECK_TOL
    assert res["max_pipe_residual"] < CHECK_TOL


def test_reversed_pipe_has_negative_flow():
    """反向录入时带符号流量为负，物理流向仍为 S→A。"""
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 2.0},
        ],
        "pipes": [{"id": "P1", "start": "A", "end": "S", "r": 1.0}],
        "min_head": 0.0,
    }
    res = solve_network(validate_payload(payload))
    assert res["flows"]["P1"] < 0
    assert res["pipe_results"][0]["direction"] == "S → A"
    assert res["heads"]["A"] == pytest.approx(26.0, abs=1e-9)


def test_loop_network_balance_and_head_law():
    res = solve_network(validate_payload(base_loop()))
    assert res["converged"] is True
    assert res["source_supply"] == pytest.approx(3.0, abs=1e-9)
    assert abs(res["balance_residual"]) < CHECK_TOL

    net = validate_payload(base_loop())
    idx = {n["id"]: i for i, n in enumerate(net.nodes)}
    H, q, r = res["heads"], res["flows"], {p["id"]: p["r"] for p in net.pipes}
    # 独立复核每根管道 H起-H终 = r·q|q|
    for p in net.pipes:
        assert H[p["start"]] - H[p["end"]] == pytest.approx(
            r[p["id"]] * q[p["id"]] * abs(q[p["id"]]), abs=CHECK_TOL)
    # 独立复核每个用水节点净流入 = 需水量
    for k, node in enumerate(net.nodes):
        if node["type"] == "demand":
            inflow = 0.0
            for p in net.pipes:
                if p["start"] == node["id"]:
                    inflow -= q[p["id"]]
                elif p["end"] == node["id"]:
                    inflow += q[p["id"]]
            assert inflow == pytest.approx(node["demand"], abs=CHECK_TOL)


def test_below_min_head_flag():
    payload = base_loop()
    payload["min_head"] = 29.0  # 所有用水节点水头约 28.7
    res = solve_network(validate_payload(payload))
    assert set(res["below_min_head"]) == {"A", "B", "C"}
    assert all(n["below_min_head"] for n in res["node_results"] if n["id"] != "S")


def test_zero_demand_nodes_allowed():
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 0.0},
            {"id": "B", "type": "demand", "demand": 3.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 2.0},
        ],
        "min_head": 0.0,
    }
    res = solve_network(validate_payload(payload))
    assert res["source_supply"] == pytest.approx(3.0, abs=1e-9)
    # B 点水头 = 30 - 1·9 - 2·9 = 3
    assert res["heads"]["B"] == pytest.approx(3.0, abs=1e-8)


# ---------------------------- 校验类用例 ---------------------------- #
@pytest.mark.parametrize("mutate,expected", [
    (lambda p: p["nodes"].__setitem__(1, {"id": "S", "type": "source", "head": 1}),
     "重复"),
    (lambda p: p["nodes"].__setitem__(1, {"id": "X", "type": "tank", "head": 1}),
     "类型"),
    (lambda p: p["nodes"].__setitem__(0, {"id": "S", "type": "source", "head": float("nan")}),
     "水头"),
    (lambda p: p["nodes"].__setitem__(1, {"id": "A", "type": "demand", "demand": -1}),
     "需水量"),
    (lambda p: p["pipes"].append({"id": "PX", "start": "S", "end": "A", "r": 1}),
     "重复"),
    (lambda p: p["pipes"][0].__setitem__("r", 0), "阻力"),
    (lambda p: p["pipes"][0].__setitem__("r", -2), "阻力"),
    (lambda p: p["pipes"].append({"id": "P9", "start": "S", "end": "S", "r": 1}),
     "自环"),
    (lambda p: p["pipes"].append({"id": "P9", "start": "S", "end": "ZZ", "r": 1}),
     "不是已定义"),
    (lambda p: p["nodes"].__setitem__(0, {"id": "S", "type": "demand", "demand": 1}),
     "水源"),
])
def test_validation_errors(mutate, expected):
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
        ],
        "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
        "min_head": 0.0,
    }
    mutate(payload)
    with pytest.raises(ValidationError) as exc:
        validate_payload(payload)
    assert expected in str(exc.value)


def test_node_and_pipe_count_limits():
    # 11 个节点
    nodes = [{"id": "S", "type": "source", "head": 1.0}]
    nodes += [{"id": f"N{i}", "type": "demand", "demand": 1.0} for i in range(10)]
    with pytest.raises(ValidationError, match="节点数"):
        validate_payload({"nodes": nodes, "pipes": [], "min_head": 0})

    # 0 根管段
    with pytest.raises(ValidationError, match="管段数"):
        validate_payload({
            "nodes": [
                {"id": "S", "type": "source", "head": 1.0},
                {"id": "A", "type": "demand", "demand": 1.0},
            ],
            "pipes": [], "min_head": 0,
        })


def test_disconnected_graph_rejected():
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
        ],
        "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
        "min_head": 0.0,
    }
    with pytest.raises(ValidationError, match="连通"):
        validate_payload(payload)


def test_non_finite_values_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {
            "nodes": [
                {"id": "S", "type": "source", "head": bad},
                {"id": "A", "type": "demand", "demand": 1.0},
            ],
            "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
            "min_head": 0.0,
        }
        with pytest.raises(ValidationError, match="有限"):
            validate_payload(payload)


# ---------------------------- API 用例 ---------------------------- #
def test_api_solve_ok():
    resp = client.post("/api/solve", json=base_loop())
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    r = data["result"]
    assert r["source_supply"] == pytest.approx(3.0, abs=1e-9)
    assert len(r["pipe_results"]) == 5


def test_api_validation_error_message():
    payload = base_loop()
    payload["pipes"][0]["r"] = -1
    resp = client.post("/api/solve", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["ok"] is False and "阻力" in data["error"]


def test_api_rejects_nan_token():
    body = '{"nodes":[{"id":"S","type":"source","head":NaN},' \
           '{"id":"A","type":"demand","demand":1}],' \
           '"pipes":[{"id":"P1","start":"S","end":"A","r":1}],"min_head":0}'
    resp = client.post("/api/solve", content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert "JSON" in resp.json()["error"]


def test_api_empty_body():
    resp = client.post("/api/solve", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_static_index_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "闭环供水管网" in resp.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"


# ---------------------------- 随机网络数值稳健性 ---------------------------- #
def test_random_connected_networks():
    rng = np.random.default_rng(0)
    n_ok = 0
    for trial in range(60):
        n = int(rng.integers(2, 11))
        edges = {tuple(sorted((i, int(rng.integers(0, i))))) for i in range(1, n)}
        target = int(rng.integers(len(edges), min(16, n * (n - 1) // 2) + 1))
        possible = list({(i, j) for i in range(n) for j in range(i + 1, n)} - edges)
        extra = min(target - len(edges), len(possible))
        perm = rng.permutation(len(possible))[:extra]
        edges.update(possible[k] for k in perm)
        src = int(rng.integers(0, n))
        nodes = [
            {"id": f"N{i}", "type": "source",
             "head": float(rng.uniform(20, 100))} if i == src
            else {"id": f"N{i}", "type": "demand",
                  "demand": float(rng.uniform(0, 5))}
            for i in range(n)
        ]
        pipes = [
            {"id": f"P{k}", "start": f"N{i}", "end": f"N{j}",
             "r": float(10 ** rng.uniform(-3, 3))}
            for k, (i, j) in enumerate(edges, 1)
        ]
        res = solve_network(validate_payload(
            {"nodes": nodes, "pipes": pipes, "min_head": -1e9}))
        assert res["max_node_residual"] < CHECK_TOL
        assert res["max_pipe_residual"] < CHECK_TOL
        assert math.isclose(res["source_supply"], res["total_demand"], abs_tol=CHECK_TOL)
        n_ok += 1
    assert n_ok == 60
