"""PalmGrad的手动力学：规定手掌、直驱关节和最多四个自由物体。

implicit_step提供Newton接触求解与implicitfast步进，state_adjoint提供状态/动作VJP。
jax_adapter把物理前向和反向接入训练；当前通过CPU JAX逐环境调度CUDA。
训练任务与数据排列分别见reference_tracking和model.hand。
"""
