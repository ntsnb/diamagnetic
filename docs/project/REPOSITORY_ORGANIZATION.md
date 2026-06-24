# 项目整理总览

本文档记录当前仓库的安全整理规则。目标不是随意搬动实验资产，而是建立清晰的分类入口，避免多版本算法、多轮实验、多份说明文档和路径改名互相干扰。

## 顶层分类

| 类别 | 当前路径 | 处理规则 |
| --- | --- | --- |
| 论文活动稿 | `bare_jrnl_new_sample4*.tex`、`bare_jrnl_new_sample4*.pdf` | 保留在根目录，避免破坏 LaTeX 编译路径、图像引用和人工查阅入口。 |
| LaTeX 编译中间文件 | `artifacts/latex/build/` | 从根目录移出。后续编译如果再生成 `.aux`、`.log`、`.fls` 等文件，应继续放入该类目录或忽略。 |
| IEEE 模板说明 | `archive/latex_templates/ieee_howto/` | `New_IEEEtran_how-to.*` 和其专用 `fig1.png` 已归档，不再与活动论文混在根目录。 |
| 源代码 | `data/`、`models/`、`train/`、`utils/`、`losses/` | 保留当前包路径。后续若移动代码文件，必须先加兼容导入或统一入口。 |
| 配置 | `configs/`、`configs/tuning/` | 使用仓库相对路径，不再写入用户机器绝对路径。 |
| 测试 | `tests/` | 作为路径整理和算法重构的回归保护。 |
| 实验日志 | `logs/` | 原位保留，不移动，不改名。通过索引归类。 |
| 模型权重 | `ckpt/`、`checkpoints/` | 原位保留，不移动，不改名。`checkpoints/` 主要承载旧 Stage III V2-V6，`ckpt/` 主要承载后续 `s3v*`。 |
| 数据 | `dataset/`、`data/` | 原位保留，不移动。配置统一指向 `dataset/...` 或其他仓库相对路径。 |
| 审计与科研记录 | `docs/audit/` | 原位保留。作为论文主张和实验结果的证据台账。 |
| Superpowers 计划与规格 | `docs/superpowers/plans/`、`docs/superpowers/specs/` | 原位保留，按日期和版本索引。 |
| 参考论文 PDF | `archive/references/papers/` | 从根目录归档。不得把这些文件当成已经引用的文献证据，论文引用仍需在 `.tex/.bib` 中真实声明。 |
| 参考论文文本提取 | `archive/references/extracted_text/` | 从根目录归档。用于人工阅读和审计，不替代原始论文 PDF。 |
| 外部工具副本 | `archive/external_sources/` | `core/`、`CLI-Anything/`、`autoresearch-master/` 已从根目录移入归档。 |
| 外部压缩归档 | `archive/external/` | `core.zip` 已移入归档目录，避免根目录被大型压缩包占据。 |
| 自动搜索结果 | `artifacts/autoresearch/` | `main_results.tsv` 和 `main_search_results.tsv` 已从根目录移入产物目录。 |

## 当前不移动的高风险资产

以下文件或目录是实验和论文证据链的一部分，目前只建立索引，不做搬迁：

- `logs/**`
- `ckpt/**`
- `checkpoints/**`
- `dataset/**`
- `data/**`
- 论文当前引用的 `.tex`、`.pdf`、`.jpg`
- 已生成且被论文或审计文档引用的图像和表格

这些路径经常出现在 `run_manifest.json`、结果 CSV、论文图表引用、实验说明文档和 checkpoint 路径中。直接搬动会造成证据断裂。

## 已完成的整理

1. 配置路径统一：旧的 `C:/Users/admin/Desktop/.../dataset/weather.csv` 和乱码旧路径已清理为 `dataset/weather.csv`。
2. 运行时路径兼容：`utils.paths.resolve_project_path` 会把仓库相对路径解析到当前项目根目录，并能从旧绝对路径中救援 `dataset/`、`configs/`、`logs/`、`ckpt/`、`checkpoints/` 等后缀。
3. 配置加载规范化：`utils.config.load_yaml` 会在加载 YAML 后规范化 `data.path`。
4. 数据读取规范化：`data.readers.load_tabular_timeseries` 会在读取文件前解析路径。
5. 输出路径稳定化：CSV 日志、TensorBoard 日志和主训练入口的 checkpoint 写入统一回当前项目根目录。
6. 低风险归档：参考 PDF/TXT、IEEE how-to 模板、LaTeX 编译中间文件、自动搜索 TSV、`core.zip` 和外部项目副本已移出根目录。

## 后续整理顺序

1. 继续补全索引，不移动证据资产。
2. 最后再考虑代码分层，例如把 Stage III 多版本代码移动到子包。代码移动必须配套导入兼容层、测试和路径检查。
