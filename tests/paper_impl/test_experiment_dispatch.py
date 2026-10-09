from pathlib import Path
import pytest
from paper_impl.experiment import plan_matrix, dispatch
from paper_impl.data import DataConfig
from paper_impl.baseline_run import normalize_baseline_config
from paper_impl.run import _normalize
import paper_impl.experiment as experiment

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/full_experiment/structural/matrix.json"


def test_selected_framework_and_official_baselines_share_identical_data_protocol():
    plan = plan_matrix(MATRIX, datasets=["lishui_city"], backbones=["itransformer"],
                       ablations=["full"], device="cpu")
    assert plan["job_count"] == 7
    reference = next(j for j in plan["jobs"] if j["kind"] == "framework")
    framework = _normalize(reference["configuration"])
    for job in plan["jobs"]:
        if job["kind"] != "baseline":
            continue
        baseline = normalize_baseline_config(job["configuration"])
        assert baseline["data"] == framework["data"]
        assert baseline["online"]["extra_delay"] == framework["online"]["extra_delay"]
        assert job["data_sha256"] == reference["data_sha256"]
        assert baseline["onenet_mode"] == "onlinetsf_cross_backbone"
        assert baseline["secondary_model"] is None
        assert baseline["online"]["period"] == 24.


def test_matrix_does_not_silently_ignore_unknown_selections():
    with pytest.raises(ValueError, match="unknown backbone"):
        plan_matrix(MATRIX, backbones=["transformer_style"])
    with pytest.raises(ValueError, match="种子 9999999 缺少配置"):
        plan_matrix(MATRIX, seeds=[9999999])
    with pytest.raises(ValueError, match="kind"):
        dispatch("legacy_proxy", {}, "unused")


def test_native_and_single_expert_selections_do_not_create_baseline_proxies():
    plan = plan_matrix(MATRIX, datasets=["tetouan"], backbones=["native"], families=["framework"],
                       ablations=["full"])
    assert plan["job_count"] == 1
    job = plan["jobs"][0]
    assert job["kind"] == "framework"
    assert DataConfig(**job["configuration"]["data"]).target_columns == ["Zone 1 Power Consumption"]


@pytest.mark.parametrize("interval", [None, 0, -1, float("nan"), float("inf"), True, "3600"])
def test_baseline_matrix_reports_missing_or_invalid_sampling_interval(monkeypatch, interval):
    original_load = experiment._load

    def load_with_interval(path):
        value = original_load(path)
        if "data" in value:
            if interval is None:
                value["data"].pop("expected_interval_seconds", None)
            else:
                value["data"]["expected_interval_seconds"] = interval
        return value

    monkeypatch.setattr(experiment, "_load", load_with_interval)
    pattern = "expected_interval_seconds.*SOLID" if interval is None else "expected_interval_seconds"
    with pytest.raises(ValueError, match=pattern):
        plan_matrix(MATRIX, datasets=["lishui_city"], backbones=["itransformer"],
                    methods=["SOLID"], families=["baseline"])
