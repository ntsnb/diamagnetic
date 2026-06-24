# 路径策略

本文档规定后续代码和配置应如何写路径，避免再次出现旧机器绝对路径、乱码路径和输出目录分叉。

## 基本规则

1. 配置文件中使用仓库相对路径，例如 `dataset/weather.csv`。
2. 不再把 `C:/Users/...` 这类用户机器绝对路径写入 YAML。
3. 训练、分析、报表脚本读取配置时应通过 `utils.config.load_yaml`。
4. 文件读取和输出路径应通过 `utils.paths.resolve_project_path` 解析。
5. 原始日志、原始 checkpoint 和原始数据不移动、不改名、不覆盖。
6. 论文当前引用的图像不移动；如果确实需要移动，必须同步更新 LaTeX 并重新编译验证。

## 推荐目录

| 产物类型 | 推荐路径 |
| --- | --- |
| 普通训练日志 | `logs/<experiment_name>/...` |
| Stage III V2-V6 旧实验日志 | `logs/load_benchmarks/stage3_v<version>/...` |
| Stage III V7 以后日志 | `logs/s3v<version>.../...` |
| 普通 checkpoint | `checkpoints/...` |
| Stage III V2-V6 checkpoint | `checkpoints/stage3_v<version>/...` |
| Stage III V7 以后 checkpoint | `ckpt/s3v<version>.../...` |
| 汇总性结果 | `results/<experiment_name>/...` |
| 论文审计记录 | `docs/audit/...` |
| LaTeX 编译中间文件 | `artifacts/latex/build/...` |
| 参考论文 PDF | `archive/references/papers/...` |
| 参考论文文本提取 | `archive/references/extracted_text/...` |
| LaTeX 模板资料 | `archive/latex_templates/...` |

## 路径兼容层

`utils.paths.resolve_project_path` 支持三类输入：

- 仓库相对路径，例如 `dataset/weather.csv`。
- 当前机器上真实存在的绝对路径。
- 历史配置中的旧绝对路径。如果旧路径已经不存在，但包含 `dataset/`、`configs/`、`logs/`、`ckpt/` 或 `checkpoints/` 等仓库后缀，函数会把这个后缀映射回当前仓库。

这层兼容是为了保护旧实验配置，不是为了鼓励继续写绝对路径。新配置仍应使用相对路径。

## 整理时的禁止事项

- 不要直接重命名 `logs/`、`ckpt/`、`checkpoints/`、`dataset/`。
- 不要把同一个实验的日志和 checkpoint 拆开移动。
- 不要为了让结果更好而修改数据路径、split、seed、metric 或模型参数。
- 不要移动论文已经引用的图片或表格，除非同步更新 LaTeX 并重新编译验证。
