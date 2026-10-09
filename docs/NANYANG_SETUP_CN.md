# nanyang 服务器部署与实验入口

本分支的实验入口为 `paper_impl`。代码、固定版本的官方模型源码、测试和冻结配置通过 GitHub 同步；原始/处理后的数据、检查点、实验结果和本地工具缓存单独保存。本文不表示已连接 nanyang 或已开始正式训练。

## 1. 拉取代码

```bash
git clone --branch codex/paper-complete-20261007 --single-branch \
  https://github.com/ntsnb/diamagnetic.git load_forecasting
cd load_forecasting
git rev-parse HEAD
```

已有该仓库时，在确认服务器工作区没有待保存修改后切换到本分支即可。不要使用 `git reset --hard` 清理已有实验文件。

## 2. 环境

本地验证环境为 Python 3.11.14、PyTorch 2.7.1+cu128。先用 `nvidia-smi` 确认服务器 GPU 和驱动，再安装相应 CUDA 构建；下例适用于兼容 CUDA 12.8 构建的驱动环境。

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-paper-implementation.txt
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

依赖从 `requirements-paper-implementation.txt` 安装。根目录 `pyproject.toml` 属于历史文稿工具，不是当前实验包；不要使用 `pip install .` 安装实验环境。所有命令从仓库根目录运行，官方骨干不在训练时下载。

## 3. 数据独立传输

在服务器新建专用数据目录，例如 `$HOME/load_forecasting_data`，将以下 13 个已处理 CSV 原样放在这个目录的同一层：

- `panama.csv`、`tetouan.csv`；
- `spain_energy_weather_clock_corrected.csv`；
- `lishui_city.csv`；
- `lishui_云和.csv`、`lishui_庆元.csv`、`lishui_景宁.csv`、`lishui_松阳.csv`、`lishui_缙云.csv`、`lishui_莲都.csv`、`lishui_遂昌.csv`、`lishui_青田.csv`、`lishui_龙泉.csv`。

对应来源路径和 SHA256 位于 `configs/full_experiment/controlled_seeds42_43/matrix.json` 的 `datasets` 字段。必须传输同一版本的文件；不要用 Excel 重存 CSV 或改变换行、编码。数据不会随 Git clone 自动下载。本步骤只为迁移已有实验输入，不代表完成了 Spain 上游插值来源核验。

## 4. 迁移冻结配置

原矩阵保留本地 Windows 路径及原始哈希，不直接在 Linux 上运行。迁移工具只使用 Python 标准库，验证数据和原配置的哈希后写入新的矩阵：

```bash
python tools/relocate_experiment_matrix.py \
  --matrix configs/full_experiment/controlled_seeds42_43/matrix.json \
  --data-root "$HOME/load_forecasting_data" \
  --output configs/full_experiment/nanyang_seeds42_43
```

迁移仅改变数据路径并更新相应配置哈希，保留种子、划分、网络、训练预算、更新时钟和全部消融设置。输出目录必须不存在；原矩阵不被覆盖。全量迁移需要上述 13 个数据文件齐全，即使随后只选择市级任务运行。

`.gitattributes` 保留冻结配置和官方源码的字节，防止 Windows/Linux 换行转换破坏校验。不要批量格式化这些已冻结文件。

## 5. 先验证，再生成计划

```bash
python -m pytest tests/paper_impl -q -p no:cacheprovider

python -m paper_impl.experiment \
  --matrix configs/full_experiment/nanyang_seeds42_43/matrix.json \
  --datasets lishui_city --backbones itransformer --ablations full \
  --seeds 42 43 --device cuda \
  --output results/nanyang_lishui_itransformer_plan
```

这一计划包含两个种子的 Full、Frozen 和六种在线基线，共 16 个独立训练任务。每个 Full 检查点还分别评价八种阶段组合和四种参数定位对照。矩阵命令不加 `--execute` 就只生成计划，不启动训练。

确定服务器资源和训练预算后，使用一个新的输出目录执行同一任务集合：

```bash
python -m paper_impl.experiment \
  --matrix configs/full_experiment/nanyang_seeds42_43/matrix.json \
  --datasets lishui_city --backbones itransformer --ablations full \
  --seeds 42 43 --device cuda --execute \
  --output results/nanyang_lishui_itransformer_run01
```

当前 5 个 epoch 是固定起始预算，是否收敛需要训练/验证曲线支持。若根据验证集调整预算，应建立新的配置版本，再冻结正式比较设置。正式结果必须同时保留配置、数据/源码哈希、检查点和诊断记录。

完整矩阵包含 5,044 个独立运行任务，建议分批调度。自动跳过已完成任务和中断后的全队列恢复尚未加入，不要把同一命令重复指向已有输出目录，也不要把不同 GPU 的独立进程指向同一结果目录。

实现与验证依据见 [新版实现说明](../README_PAPER_IMPLEMENTATION.md) 和 [控制实验验收记录](audit/controlled_comparisons_20261009/IMPLEMENTATION_RECORD.md)。
