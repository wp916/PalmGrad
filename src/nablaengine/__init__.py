"""PalmGrad 使用的手动力学及 NablaEngine 共享 CUDA 实现。

手专用接口位于 model.hand、hand_engine；g1_engine 保留被复用的上游底层代码。
训练由 hand_training 组织，本包负责动力学及其导数。
"""

__version__ = "0.1.0"
