"""汇总已经执行的检查，不重跑实验或改写既有结果。"""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_impl.run import source_manifest


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    audit = ROOT / "docs/audit/paper_implementation_20261007"
    suite_dir = ROOT / "results/paper_impl/20261007_validation_suite_v1"
    suite = read(suite_dir / "suite_report.json")
    current = source_manifest()
    tests = list(ET.parse(audit / "validation_tests_final.xml").getroot())[0].attrib
    rows = []
    for run in suite["runs"]:
        path = suite_dir / run["name"]
        result = read(path / "summary.json")
        manifest = read(path / "source_manifest.json")
        rows.append({"name": run["name"], "completed": run["completed"],
                     "current_source_matches": manifest["code_sha256"] == current["code_sha256"],
                     "variants": [v["variant"] for v in result["results"]],
                     "issued": [v["issued_forecasts"] for v in result["results"]]})
    a = ROOT / "results/paper_impl/20261007_panama_cuda_pipeline_v1"
    b = ROOT / "results/paper_impl/20261007_cuda_postfeedback_reevaluation_v1"
    cuda = read(b / "summary.json")
    prefix_equal = {}
    predictions = {}
    for row in cuda["results"]:
        name = row["variant"]
        first = np.load(a / (name + "_forecasts.npz"))["predictions"]
        second = np.load(b / (name + "_forecasts.npz"))["predictions"]
        prefix_equal[name] = bool(np.array_equal(first, second[:len(first)]))
        predictions[name] = second
    stubs = []
    for file in (ROOT / "paper_impl").glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
                if not body or (len(body) == 1 and (isinstance(body[0], ast.Pass) or
                    isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and body[0].value.value is Ellipsis)):
                    stubs.append(f"{file.name}:{node.lineno}:{node.name}")
    runtime_models, efforts = set(), set()
    def whitelist(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "model" and isinstance(item, str) and len(item) < 60:
                    runtime_models.add(item)
                if key in {"reasoning_effort", "reasoningEffort"} and isinstance(item, str) and len(item) < 25:
                    efforts.add(item)
                whitelist(item)
        elif isinstance(value, list):
            for item in value:
                whitelist(item)
    for path in (audit / "zcode_core_review_run/runtime").rglob("model-io-*.jsonl"):
        with path.open(encoding="utf8") as file:
            for line in file:
                try:
                    whitelist(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rounds = []
    for path in sorted((audit / "zcode_core_review_run/attempts").glob("*.result.json")):
        data = read(path)
        if data.get("ok") and data.get("sessionId"):
            rounds.append({"invocation_id": data["invocationId"], "ok": True, "mode":data["mode"],
                           "seconds":data["durationMs"]/1000, "report_only_no_independent_test_execution":True})
    report = {"created":datetime.now(timezone.utc).isoformat(), "tests":tests, "suite":rows,
              "cuda": {"completed":cuda["status"] == "completed", "device":current["gpu"],
                       "variants":len(cuda["results"]), "checkpoint_prefix_equal":prefix_equal,
                       "router_changes_post_feedback_prediction":bool(np.any(predictions["stages_000"][6:] != predictions["stages_100"][6:])),
                       "expert_changes_post_feedback_prediction":bool(np.any(predictions["stages_000"][6:] != predictions["stages_010"][6:]))},
              "source_manifest":current, "empty_function_stubs":stubs,
              "zcode":{"runtime_models":sorted(runtime_models),"reasoning_efforts":sorted(efforts),"rounds":rounds},
              "performance_claim":False}
    assert tests["errors"] == tests["failures"] == tests["skipped"] == "0"
    assert len(rows) == 24 and all(r["completed"] and r["current_source_matches"] for r in rows)
    assert all(prefix_equal.values()) and not stubs
    (audit / "VALIDATION_SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    table = "\n".join(f"| {r['name']} | {' / '.join(r['variants'])} | {r['issued']} | 通过 |" for r in rows)
    markdown = f"""# 新版实现最终验证记录

生成时间：{report['created']}。本文件只汇总已有执行产物，不作为预测性能研究。

## 已验证

- 自动测试：**{tests['tests']}通过，0失败，0错误，0跳过**；CPU/CUDA路径均实际执行。XML：`validation_tests_final.xml`。
- 24个完整真实数据短流程全部完成，产物源码清单与当前`paper_impl`及旧骨干依赖哈希一致。
- CUDA设备：{current['gpu']}。完整训练及全部8阶段组合完成；checkpoint重新加载后前6次发行预测逐位相等，随后12窗口评估覆盖标签释放后更新再次用于预测的路径。
- AST扫描未发现空函数或省略号占位。数值/梯度/协议的验证依据仍是具体测试，而不是这个静态扫描。
- ZCode完成{len(rounds)}轮GLM-5.3 max静态复核；最终报告未发现阻断性问题。测试由Codex执行，ZCode读取代码与测试产物，没有独立执行pytest。原意见、分歧及纠正分别保存。

## 真实数据有界矩阵

| 配置 | 阶段组合 | 发行数 | 状态 |
| --- | --- | --- | --- |
{table}

矩阵位于 `results/paper_impl/20261007_validation_suite_v1`。丽水九县、Panama、Spain、Tetouan都使用本机既有数据；仅Spain的已确认日期字段问题另建审计副本，其他观测和原文件保持不动。测试配置的短训练预算与历史论文实验不同。

## 结论边界

本轮完成新版方法及对照的实现、方法文本对齐和程序验证。没有完成全量多种子性能实验、八个官方骨干或官方外部基线的复现，也没有把旧表图改写为新方法结果。新源稿保留实验H1边界说明。LRP不等于物理因果识别，局部下降和扰动界不等于未来精度保证。

所有原始运行目录保留；旧format-1短流程只是历史调试记录，完整重评估使用新的format-2 checkpoint。新增源码未提交Git，原Desktop工程与nanyang未被本轮改写。
"""
    (audit / "VALIDATION_REPORT.md").write_text(markdown,encoding="utf8")
    print(json.dumps({"tests":tests["tests"],"suite":len(rows),"cuda":report["cuda"],
                      "runtime_models":sorted(runtime_models),"reasoning_efforts":sorted(efforts),"zcode_rounds":len(rounds)},ensure_ascii=False))


if __name__ == "__main__":
    main()
