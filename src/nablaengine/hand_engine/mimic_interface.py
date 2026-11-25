"""手任务接入NablaMimic的状态/动作接缝，不包含任务奖励或训练算法。"""
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp


class NativeHandData(NamedTuple):
    qpos:object
    qvel:object
    qacc_warmstart:object


class FixedHandContext(NamedTuple):
    friction:object
    com:object


class HandMimicState(NamedTuple):
    dx:NativeHandData
    frame:object       # 本阶段是控制步计数，尚不是参考动作帧。
    ep:object
    prev_a:object      # 原策略动作；不存PD目标冒充归一化动作。
    xi:FixedHandContext


class HandStateOrder:
    """源模型状态/执行器顺序 ↔ hand canonical，纯JAX gather保留自动求导。"""
    def __init__(self,layout):
        self.layout=layout
        def permutation(length,blocks,start,size):
            ids=[]
            for b in blocks:ids.extend(range(getattr(b,start),getattr(b,start)+getattr(b,size)))
            ids=np.array(ids,dtype=np.int32)
            if not np.array_equal(np.sort(ids),np.arange(length)):raise ValueError('layout未完整覆盖源状态或存在重复地址')
            ids.setflags(write=False);return ids
        self.qpos=permutation(layout.source_nq,layout.blocks,'source_qpos_start','nq')
        self.qvel=permutation(layout.source_nv,layout.blocks,'source_qvel_start','nv')
        ctrl=np.array([a.source_control_address for a in layout.actuators],dtype=np.int32)
        if not np.array_equal(np.sort(ctrl),np.arange(layout.hand_dofs)):raise ValueError('执行器控制地址必须为完整排列')
        ctrl.setflags(write=False);self.ctrl=ctrl
        self.inv_qpos=np.argsort(self.qpos);self.inv_qvel=np.argsort(self.qvel);self.inv_ctrl=np.argsort(self.ctrl)
        for a in (self.inv_qpos,self.inv_qvel,self.inv_ctrl):a.setflags(write=False)

    @staticmethod
    def _take(value,ids):
        if value.ndim<1 or value.shape[-1]!=len(ids):raise ValueError('状态/控制最后一维与排列不匹配')
        return jnp.take(value,jnp.asarray(ids),axis=-1)

    def pack_qpos(self,x):return self._take(x,self.qpos)
    def pack_qvel(self,x):return self._take(x,self.qvel)
    def pack_ctrl(self,x):return self._take(x,self.ctrl)
    def unpack_qpos(self,x):return self._take(x,self.inv_qpos)
    def unpack_qvel(self,x):return self._take(x,self.inv_qvel)
    def unpack_ctrl(self,x):return self._take(x,self.inv_ctrl)


class NativeHandPhysics:
    """只在物理边界转换排列；对外q/v/w/ctrl均用源模型顺序。"""
    def __init__(self,physics,order):
        if physics.layout is not order.layout:raise ValueError('物理与排列必须使用同一layout实例')
        self.physics=physics;self.order=order;self.nworld=physics.nworld
        self.info=dict(physics.info,state_order='native_model',control_order='native_actuator')

    def step(self,q,v,w,u,friction,com):
        o=self.order
        q,v,w=self.physics.step(o.pack_qpos(q),o.pack_qvel(v),o.pack_qvel(w),o.pack_ctrl(u),friction,com)
        return o.unpack_qpos(q),o.unpack_qvel(v),o.unpack_qvel(w)


class HandActionMap:
    """沿用NablaMimic ctrl_of：clip(base + scale * action)，参数按源执行器顺序。

    不自行从人体参数推算手的动作范围；调用者必须明确给出各关节映射。
    这里是外部命令裁剪，不修改引擎内部动力学或隐式加入力矩饱和。
    """
    def __init__(self,base,scale,lower,upper):
        values=[np.array(x,dtype=np.float64,copy=True) for x in (base,scale,lower,upper)]
        shape=values[0].shape
        if len(shape)!=1 or shape[0]<1 or any(x.shape!=shape or not np.isfinite(x).all() for x in values):raise ValueError('动作映射需要同长有限一维数组')
        if np.any(values[1]<=0) or np.any(values[2]>=values[3]):raise ValueError('scale必须正，lower必须小于upper')
        for x in values:x.setflags(write=False)
        self.base,self.scale,self.lower,self.upper=values;self.nu=shape[0]

    def __call__(self,action):
        if action.ndim<1 or action.shape[-1]!=self.nu:raise ValueError('策略动作维度不匹配')
        return jnp.clip(jnp.asarray(self.base)+jnp.asarray(self.scale)*action,jnp.asarray(self.lower),jnp.asarray(self.upper))


class HandMimicInterface:
    """可被原sim.vmapped调用的真实物理接缝；任务方法尚未接通则明确拒绝。

    当前只支持固定规定手掌/上下文，不把控制计数当成参考动作帧。
    使用原生模型顺序存dx，便于后续原生模型运动学/观测与状态回填。
    """
    def __init__(self,physics,action_map,*,control_dt):
        if action_map.nu!=physics.layout.hand_dofs:raise ValueError('动作映射与手关节数不一致')
        if not np.isfinite(control_dt) or not np.isclose(control_dt,physics.info['control_dt'],rtol=0,atol=1e-14):raise ValueError('环境控制周期必须等于物理步长乘substeps')
        kinds={a.source_kind for a in physics.layout.actuators}
        if kinds!={physics.mode}:raise ValueError('模型执行器类型与物理命令模式不一致')
        if physics.mode=='position_pd':
            if not np.array_equal(physics.kp,[a.kp for a in physics.layout.actuators]) or not np.array_equal(physics.kv,[a.kv for a in physics.layout.actuators]):raise ValueError('PD增益需按canonical关节顺序与模型一致')
        self.layout=physics.layout;self.order=HandStateOrder(self.layout);self.action_map=action_map;self.dt=float(control_dt)
        self.native_physics=NativeHandPhysics(physics,self.order);self.warp={physics.nworld:self.native_physics}

    def bind(self,tables):
        if tables is not None:raise ValueError('参考动作表尚未接通')
        return self

    def initial_state(self,qpos,qvel,*,warm=None):
        qpos=np.asarray(qpos,dtype=np.float64);qvel=np.asarray(qvel,dtype=np.float64);B=self.native_physics.nworld
        if qpos.shape!=(B,self.layout.source_nq) or qvel.shape!=(B,self.layout.source_nv):raise ValueError('初始原生状态batch或维度不匹配')
        for q,v in zip(qpos,qvel):self.layout.pack(q,v)
        warm=np.zeros_like(qvel) if warm is None else np.asarray(warm,dtype=np.float64)
        if warm.shape!=qvel.shape or not np.isfinite(warm).all():raise ValueError('warm必须是原生自由度顺序的有限数组')
        return HandMimicState(NativeHandData(jnp.asarray(qpos),jnp.asarray(qvel),jnp.asarray(warm)),
                              jnp.zeros(B,jnp.int32),jnp.zeros(B,jnp.int32),jnp.zeros((B,self.layout.hand_dofs),jnp.float64),
                              FixedHandContext(jnp.ones(B,jnp.float64),jnp.zeros((B,3),jnp.float64)))

    def ctrl_of(self,action,frame=None):return self.action_map(action)

    def finish_physics(self,state,action,qpos,qvel,warm):
        return state._replace(dx=NativeHandData(qpos,qvel,warm),frame=state.frame+1,ep=state.ep+1,prev_a=jnp.asarray(action,dtype=state.prev_a.dtype))

    def _task_not_ready(self,*args,**kwargs):
        raise NotImplementedError('手操作任务的观测/奖励/终止/reset尚未配置；当前只开放物理接缝验证')

    reset=policy_obs=critic_obs=reward=terminated=clip_ended=hazard_features=_task_not_ready
