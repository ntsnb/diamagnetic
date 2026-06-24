# Models Directory Guide

本目录保存预测模型、表示学习模块和 Stage III 多版本模型组件。

## 基础模型

| 文件 | 用途 |
| --- | --- |
| `factor_mechanism.py` | 主预测模型，包括因子机制、专家和 VAE 相关逻辑。 |
| `factor_mechanism_v2.py` | Stage III 修复路径使用的确定性 VAE eval 包装。 |
| `invariant_learning.py` | 不变学习相关模型。 |
| `simple_encoders.py` | 简单 MLP 编码器和线性回归头。 |
| `attention_blocks.py` | 注意力模块。 |
| `lrp.py` | 相关性传播和解释性工具。 |

## Stage III 组件

| 版本族 | 文件 |
| --- | --- |
| Legacy | `adaptation.py` |
| V2 | `adaptation_v2.py`、`aligned_representation_v2.py` |
| V7/V9 | `stage3_alignment_v7.py`、`stage3_alignment_v9.py`、`stage3_residual_teacher_v9.py` |
| V10-V12 | `stage3_alignment_v10.py`、`stage3_alignment_v11.py`、`stage3_alignment_v12.py`、`stage3_residual_teacher_v10.py`、`stage3_residual_teacher_v11.py`、`stage3_residual_teacher_v12.py` |
| V12B-V12D | `stage3_residual_teacher_v12b.py`、`stage3_residual_teacher_v12c.py`、`stage3_residual_teacher_v12d.py` |
| V13-V14 | `stage3_residual_teacher_v13.py`、`stage3_residual_teacher_v14.py` |

## 注意事项

不要把旧版 Stage III 文件删除或覆盖。它们对应已有实验记录和失败诊断，是论文科研完整性证据链的一部分。
