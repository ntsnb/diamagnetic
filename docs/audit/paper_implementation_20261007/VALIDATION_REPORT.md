# 新版实现最终验证记录

生成时间：2026-10-07T06:10:08.932823+00:00。本文件只汇总已有执行产物，不作为预测性能研究。

## 已验证

- 自动测试：**258通过，0失败，0错误，0跳过**；CPU/CUDA路径均实际执行。XML：`validation_tests_final.xml`。
- 24个完整真实数据短流程全部完成，产物源码清单与当前`paper_impl`及旧骨干依赖哈希一致。
- CUDA设备：NVIDIA GeForce RTX 5060。完整训练及全部8阶段组合完成；checkpoint重新加载后前6次发行预测逐位相等，随后12窗口评估覆盖标签释放后更新再次用于预测的路径。
- AST扫描未发现空函数或省略号占位。数值/梯度/协议的验证依据仍是具体测试，而不是这个静态扫描。
- ZCode完成3轮GLM-5.3 max静态复核；最终报告未发现阻断性问题。测试由Codex执行，ZCode读取代码与测试产物，没有独立执行pytest。原意见、分歧及纠正分别保存。

## 真实数据有界矩阵

| 配置 | 阶段组合 | 发行数 | 状态 |
| --- | --- | --- | --- |
| lishui_云和 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_庆元 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_景宁 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_松阳 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_缙云 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_莲都 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_遂昌 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_青田 | stages_000 / stages_111 | [12, 12] | 通过 |
| lishui_龙泉 | stages_000 / stages_111 | [12, 12] | 通过 |
| panama | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_autoformer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_crossformer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_fedformer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_gradient_only | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_informer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_inverted_transformer | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_itransformer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_patch_transformer | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_patchtst_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_pyraformer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_shared_generator | stages_000 / stages_111 | [12, 12] | 通过 |
| panama_transformer_style | stages_000 / stages_111 | [12, 12] | 通过 |
| spain | stages_000 / stages_111 | [12, 12] | 通过 |
| tetouan | stages_000 / stages_111 | [12, 12] | 通过 |

矩阵位于 `results/paper_impl/20261007_validation_suite_v1`。丽水九县、Panama、Spain、Tetouan都使用本机既有数据；仅Spain的已确认日期字段问题另建审计副本，其他观测和原文件保持不动。测试配置的短训练预算与历史论文实验不同。

## 结论边界

本轮完成新版方法及对照的实现、方法文本对齐和程序验证。没有完成全量多种子性能实验、八个官方骨干或官方外部基线的复现，也没有把旧表图改写为新方法结果。新源稿保留实验H1边界说明。LRP不等于物理因果识别，局部下降和扰动界不等于未来精度保证。

所有原始运行目录保留；旧format-1短流程只是历史调试记录，完整重评估使用新的format-2 checkpoint。新增源码未提交Git，原Desktop工程与nanyang未被本轮改写。
