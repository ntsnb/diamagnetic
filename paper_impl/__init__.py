"""论文十月版本：专家分工与延迟反馈下的三阶段在线预测。"""

# cuBLAS reads this setting when its first handle is created. Setting it only
# when an individual runner starts is too late in a shared Python process that
# has already evaluated another CUDA model. This import does not import torch
# or initialize CUDA; an explicit caller setting always takes precedence.
import os as _os

_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
del _os
