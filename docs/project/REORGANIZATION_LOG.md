# 项目整理日志

## 2026-06-07 第一批整理：路径稳定化

目标：先解决路径混乱和版本查找困难，不移动原始实验资产。

### 代码改动

- `utils/paths.py`
  - 新增 `PROJECT_ROOT` 和 `project_root()`。
  - 新增 `resolve_project_path()`，支持仓库相对路径、真实绝对路径和历史旧绝对路径后缀救援。
  - `ensure_dir()` 改为基于当前项目根目录解析路径。
- `utils/config.py`
  - `load_yaml()` 会先解析配置文件路径。
  - 加载后自动规范化 `data.path`。
- `data/readers.py`
  - 读取 Excel、XML、CSV 前统一解析数据路径。
- `utils/logging.py`
  - CSV 日志写入前统一解析路径并创建父目录。
- `utils/tb_logger.py`
  - TensorBoard 日志目录统一解析到当前项目根目录。
- `train/run_experiment.py`
  - 主训练入口的 checkpoint 写入路径统一解析到项目根目录。
- `train/prepare_lishui_dataset.py`
  - 默认原始数据目录从用户机器绝对路径改为 `dataset/raw/lishui`。
  - `--raw-root` 和 `--output-dir` 均通过统一路径层解析。
- `tools/audit_project_paths.py`
  - 新增只读路径审计工具，用于检查配置中的旧绝对路径和不可解析数据路径。

### 配置清理

以下配置的 `data.path` 从旧绝对路径或乱码绝对路径改为 `dataset/weather.csv`：

- `configs/base.yaml`
- `configs/best_online.yaml`
- `configs/best_lishui_online.yaml`
- `configs/best_load_online.yaml`
- `configs/tuning/load_candidate_a.yaml`
- `configs/tuning/load_candidate_a2.yaml`
- `configs/tuning/load_candidate_b.yaml`
- `configs/tuning/load_candidate_c_revin.yaml`
- `configs/tuning/load_candidate_d_revin_big.yaml`
- `configs/tuning/load_candidate_e_revin_mixed.yaml`
- `configs/tuning/load_candidate_f_revin_small.yaml`
- `configs/tuning/load_candidate_g_itransformer.yaml`
- `configs/tuning/load_candidate_h_online.yaml`
- `configs/tuning/load_candidate_i_shifted.yaml`
- `configs/tuning/lishui_candidate_j_mlp_shifted.yaml`
- `configs/tuning/lishui_candidate_k_mlp_shifted.yaml`
- `configs/tuning/lishui_candidate_l_vae_shifted.yaml`
- `configs/tuning/lishui_candidate_m_stage3strong.yaml`
- `configs/tuning/lishui_candidate_n_crossformer_shifted.yaml`
- `configs/tuning/lishui_candidate_o_crossformer_plus.yaml`
- `configs/tuning/lishui_candidate_p_crossformer_mlp_stage3plus.yaml`
- `configs/tuning/lishui_candidate_q_crossformer_mlp_huber.yaml`
- `configs/tuning/lishui_candidate_r_crossformer_vae_stage3peak.yaml`

### 验证结果

- `C:\Users\admin\anaconda3\python.exe tools\audit_project_paths.py --strict`
  - `legacy_absolute_path_hits=0`
  - `unresolved_data_paths=0`
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_project_paths.py tests\test_stage3_v15_residual_target_predictability.py tests\test_stage3_v14_teacher_protocol.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v12e_label_availability.py -q`
  - `31 passed`
- `C:\Users\admin\anaconda3\python.exe -m py_compile ...`
  - 修改过的 Python 文件语法检查通过。
- `C:\Users\admin\anaconda3\python.exe tools\audit_latex_assets.py`
  - `checked_tex_files=5`
  - `missing_graphics=0`

## 2026-06-07 第二批整理：低风险归档

目标：让根目录更整齐，同时不移动原始数据、日志、checkpoint、论文主稿、论文 PDF 和论文当前引用图像。

### 已归档内容

- LaTeX 编译中间文件移动到 `artifacts/latex/build/`：
  - `bare_jrnl_new_sample4*.aux`
  - `bare_jrnl_new_sample4*.log`
  - `bare_jrnl_new_sample4*.fdb_latexmk`
  - `bare_jrnl_new_sample4*.fls`
  - `bare_jrnl_new_sample4*.xdv`
- IEEE how-to 模板移动到 `archive/latex_templates/ieee_howto/`：
  - `New_IEEEtran_how-to.*`
  - `fig1.png`
- 参考论文 PDF 移动到 `archive/references/papers/`。
- 参考论文文本提取移动到 `archive/references/extracted_text/`。
- 大型外部压缩包 `core.zip` 移动到 `archive/external/`。
- 外部项目副本移动到 `archive/external_sources/`：
  - `core/`
  - `CLI-Anything/`
  - `autoresearch-master/`
- 自动搜索 TSV 移动到 `artifacts/autoresearch/`：
  - `main_results.tsv`
  - `main_search_results.tsv`

### 新增目录指南

- `configs/README.md`
- `train/README.md`
- `models/README.md`
- `artifacts/autoresearch/README.md`
- `archive/external_sources/README.md`
- `tools/audit_repository_layout.py`

### 验证结果

- `C:\Users\admin\anaconda3\python.exe tools\audit_project_paths.py --strict`
  - `legacy_absolute_path_hits=0`
  - `unresolved_data_paths=0`
- `C:\Users\admin\anaconda3\python.exe tools\audit_latex_assets.py`
  - `checked_tex_files=5`
  - `missing_graphics=0`
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_project_paths.py tests\test_stage3_v15_residual_target_predictability.py tests\test_stage3_v14_teacher_protocol.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v12e_label_availability.py -q`
  - `31 passed`
- 根目录扫描确认不再残留 `.aux`、`.log`、`.fdb_latexmk`、`.fls`、`.xdv`、`.synctex.gz` 或 `core.zip`。

## 2026-06-07 第三批整理：外部树和顶层搜索结果归档

目标：移除根目录中的外部项目副本和自动搜索 TSV，减少全仓扫描噪声。

### 已归档内容

- `core/` 移动到 `archive/external_sources/core/`。
- `CLI-Anything/` 移动到 `archive/external_sources/CLI-Anything/`。
- `autoresearch-master/` 移动到 `archive/external_sources/autoresearch-master/`。
- `main_results.tsv` 移动到 `artifacts/autoresearch/main_results.tsv`。
- `main_search_results.tsv` 移动到 `artifacts/autoresearch/main_search_results.tsv`。

### 代码路径更新

- `train/run_ablation_suite.py` 的默认结果路径改为 `artifacts/autoresearch/main_results.tsv`。
- `train/run_autoresearch_main.py` 的默认搜索结果路径改为 `artifacts/autoresearch/main_search_results.tsv`。
- `train/run_autoresearch_main.py` 内部调用 `run_ablation_suite` 时也写入 `artifacts/autoresearch/main_results.tsv`。

### 验证结果

- `C:\Users\admin\anaconda3\python.exe tools\audit_repository_layout.py --strict`
  - `layout_problems=0`
- `C:\Users\admin\anaconda3\python.exe tools\audit_project_paths.py --strict`
  - `legacy_absolute_path_hits=0`
  - `unresolved_data_paths=0`
- `C:\Users\admin\anaconda3\python.exe tools\audit_latex_assets.py`
  - `missing_graphics=0`
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_project_paths.py tests\test_stage3_v15_residual_target_predictability.py tests\test_stage3_v14_teacher_protocol.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v12e_label_availability.py -q`
  - `31 passed`

### 保留原位内容

- `logs/`
- `ckpt/`
- `checkpoints/`
- `dataset/`
- `data/`
- 论文活动 `.tex` 和 `.pdf`
- `fig1.jpg` 和 `fig2.jpg`

`fig1.jpg` 和 `fig2.jpg` 仍被 `bare_jrnl_new_sample4*.tex` 直接引用，因此没有移动。
