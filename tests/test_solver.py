"""求解器与 API 的端到端测试（pytest）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    CHECK_TOL,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    STATUS_NOT_CONVERGED,
    SolverNotConverged,
    ValidationError,
    analyze_pipe_shutdowns,
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


# ---------------------------- 逐根停管分析 ---------------------------- #
def shutdown_tree():
    """树状网：S-A-B 主干，B-C 支段，A-D 支段；每根都是桥。"""
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 40.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 2.0},
            {"id": "C", "type": "demand", "demand": 4.0},
            {"id": "D", "type": "demand", "demand": 3.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 0.1},
            {"id": "P2", "start": "A", "end": "B", "r": 0.5},
            {"id": "P3", "start": "B", "end": "C", "r": 0.5},
            {"id": "P4", "start": "A", "end": "D", "r": 0.2},
        ],
        "min_head": 20.0,
    }


def shutdown_loop():
    """环 S-A-B-C-S 加末端分支 A-D（P5 是唯一的桥）。"""
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 2.0},
            {"id": "C", "type": "demand", "demand": 1.0},
            {"id": "D", "type": "demand", "demand": 2.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 0.05},
            {"id": "P2", "start": "A", "end": "B", "r": 0.5},
            {"id": "P3", "start": "B", "end": "C", "r": 0.5},
            {"id": "P4", "start": "C", "end": "S", "r": 0.05},
            {"id": "P5", "start": "A", "end": "D", "r": 0.3},
        ],
        "min_head": 20.0,
    }


def test_shutdown_tree_all_disconnected_sorted_by_demand():
    res = analyze_pipe_shutdowns(validate_payload(shutdown_tree()))
    scenarios = res["scenarios"]

    assert res["disconnected_count"] == 4
    assert res["connected_count"] == 0
    assert res["not_converged_count"] == 0
    assert [s["pipe_id"] for s in scenarios] == ["P1", "P2", "P3", "P4"]
    # 复核失供需水量严格降序（P1: 1+2+4+3=10，P2: B+C=6，P3: C=4，P4: D=3）
    demands = [s["unserved_demand"] for s in scenarios]
    assert demands == [10.0, 6.0, 4.0, 3.0]
    assert demands == sorted(demands, reverse=True)

    # 断联情景不给水头
    p1 = scenarios[0]
    assert p1["status"] == STATUS_DISCONNECTED
    assert p1["unserved_nodes"] == ["A", "B", "C", "D"]
    assert "min_demand_head" not in p1
    assert "heads" not in p1

    # 各情景风险序连续
    assert [s["rank"] for s in scenarios] == [1, 2, 3, 4]


def test_shutdown_loop_mixed_statuses_and_new_below():
    res = analyze_pipe_shutdowns(validate_payload(shutdown_loop()))
    scenarios = res["scenarios"]

    assert res["disconnected_count"] == 1
    assert res["connected_count"] == 4
    assert res["not_converged_count"] == 0

    # 唯一断联情景（P5，失供 D=2.0）排最前
    first = scenarios[0]
    assert first["status"] == STATUS_DISCONNECTED
    assert first["pipe_id"] == "P5"
    assert first["unserved_nodes"] == ["D"]
    assert first["unserved_demand"] == 2.0

    # 其余连通情景按最低用水节点水头升序
    connected = [s for s in scenarios if s["status"] == STATUS_CONNECTED]
    heads = [s["min_demand_head"] for s in connected]
    assert heads == sorted(heads)
    for s in connected:
        assert s["min_demand_head"] == s["heads"][s["min_demand_node"]]

    # 基线全部合格；停 P1 后 A/B/D 低于限值且全部为“新增”
    assert res["baseline"]["below_min_head"] == []
    p1 = next(s for s in scenarios if s["pipe_id"] == "P1")
    assert set(p1["below_limit_nodes"]) == {"A", "B", "D"}
    assert set(p1["new_below_nodes"]) == {"A", "B", "D"}


def test_shutdown_new_below_relative_to_baseline():
    """基线已有的低水头节点不计入“新增”。"""
    payload = shutdown_loop()
    payload["min_head"] = 29.0  # 基线下 A、D 等已低于限值
    res = analyze_pipe_shutdowns(validate_payload(payload))
    baseline_below = set(res["baseline"]["below_min_head"])
    for s in res["scenarios"]:
        if s["status"] != STATUS_CONNECTED:
            continue
        assert not (set(s["new_below_nodes"]) & baseline_below)
        assert set(s["new_below_nodes"]).issubset(set(s["below_limit_nodes"]))


def test_shutdown_ties_keep_pipe_entry_order():
    """失供量相同的断联情景同分，保持管段录入顺序。"""
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "J", "type": "demand", "demand": 0.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
        ],
        # 树：S-J 主干（失供 2.0），J-A、J-B 两个分支桥失供量同为 1.0
        "pipes": [
            {"id": "P1", "start": "S", "end": "J", "r": 1.0},
            {"id": "P2", "start": "J", "end": "A", "r": 1.0},
            {"id": "P3", "start": "J", "end": "B", "r": 1.0},
        ],
        "min_head": 0.0,
    }
    res = analyze_pipe_shutdowns(validate_payload(payload))
    scenarios = res["scenarios"]
    assert [s["pipe_id"] for s in scenarios] == ["P1", "P2", "P3"]
    assert all(s["status"] == STATUS_DISCONNECTED for s in scenarios)
    assert scenarios[1]["unserved_nodes"] == ["A"]
    assert scenarios[2]["unserved_nodes"] == ["B"]
    assert scenarios[1]["unserved_demand"] == scenarios[2]["unserved_demand"] == 1.0


def test_shutdown_single_not_converged_isolated_and_last(monkeypatch):
    """某个连通情景不收敛：单独标明、排最后，其余情景照常。"""
    from app import solver as solver_mod

    payload = shutdown_loop()
    net = validate_payload(payload)
    original = solver_mod.solve_network
    state = {"calls": 0}

    def patched(network):
        state["calls"] += 1
        # 第 1 次是基线；P5 在最前断联不触发求解；
        # 随后依次是 P1、P2、P3、P4 的连通求解，让 P3（第 4 次）不收敛
        if state["calls"] == 4:
            raise SolverNotConverged("模拟 P3 情景不收敛")
        return original(network)

    monkeypatch.setattr(solver_mod, "solve_network", patched)
    res = solver_mod.analyze_pipe_shutdowns(net)

    assert res["connected_count"] == 3
    assert res["not_converged_count"] == 1
    last = res["scenarios"][-1]
    assert last["pipe_id"] == "P3"
    assert last["status"] == STATUS_NOT_CONVERGED
    assert "不收敛" in last["error"]
    assert "heads" not in last
    # 其余情景不受影响
    assert all(s["status"] != STATUS_NOT_CONVERGED for s in res["scenarios"][:-1])
    # 断联仍排在最前
    assert res["scenarios"][0]["status"] == STATUS_DISCONNECTED


def test_shutdown_zero_demand_unserved_listed():
    """零需水量的失供节点仍须列出，只是失供需水量合计为 0。"""
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 0.0},
            {"id": "B", "type": "demand", "demand": 0.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
        ],
        "min_head": 0.0,
    }
    res = analyze_pipe_shutdowns(validate_payload(payload))
    p1 = res["scenarios"][0]
    assert p1["status"] == STATUS_DISCONNECTED
    assert set(p1["unserved_nodes"]) == {"A", "B"}
    assert p1["unserved_demand"] == 0.0


def test_api_shutdown_analysis_ok():
    resp = client.post("/api/shutdown-analysis", json=shutdown_loop())
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    r = data["result"]
    assert len(r["scenarios"]) == 5
    assert r["baseline"]["converged"] is True
    assert {s["pipe_id"] for s in r["scenarios"]} == {"P1", "P2", "P3", "P4", "P5"}
    # 基线字段与 /api/solve 一致
    assert "heads" in r["baseline"] and "flows" in r["baseline"]


def test_api_shutdown_analysis_validation_error():
    payload = shutdown_tree()
    payload["pipes"][0]["r"] = -1
    resp = client.post("/api/shutdown-analysis", json=payload)
    assert resp.status_code == 422
    assert "阻力" in resp.json()["error"]


def test_api_shutdown_analysis_empty_body():
    resp = client.post("/api/shutdown-analysis", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_api_shutdown_analysis_rejects_nan_token():
    body = '{"nodes":[{"id":"S","type":"source","head":NaN},' \
           '{"id":"A","type":"demand","demand":1}],' \
           '"pipes":[{"id":"P1","start":"S","end":"A","r":1}],"min_head":0}'
    resp = client.post("/api/shutdown-analysis", content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert "JSON" in resp.json()["error"]


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
