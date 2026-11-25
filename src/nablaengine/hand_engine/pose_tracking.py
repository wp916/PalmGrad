"""最小单物体静态目标位姿任务；复用 Mimic 接缝，不修改动力学或 rollout。"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from .mimic_interface import HandMimicInterface, NativeHandData, FixedHandContext


class PoseTrackingState(NamedTuple):
    dx: NativeHandData
    frame: object
    ep: object
    prev_a: object
    xi: FixedHandContext
    target_pos: object
    target_quat: object
    key: object  # 原Mimic重放采样会刷新此key；当前观测仍无噪声。


def rotation6(q):
    """wxyz 单位四元数的前两列旋转矩阵，避免 q/-q 观测跳变。"""
    w,x,y,z=q
    return jnp.array([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y),
                      2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)])


class SingleObjectPoseTracking(HandMimicInterface):
    """世界系物体根位姿跟踪；单帧观测，无参考动作表。

    reset(key, frame) 保留原训练调用签名；静态目标不使用参考帧参数。
    物理初态固定；仅随机目标平移，避免本阶段引入未经验证的接触初态采样。
    """
    def __init__(self,physics,action_map,*,control_dt,reset_qpos,reset_qvel,
                 target_pos,target_quat,position_scale=.02,orientation_scale=.5,
                 action_rate_weight=.01,max_distance=.25,target_jitter=0.):
        super().__init__(physics,action_map,control_dt=control_dt)
        objects=[b for b in self.layout.blocks if b.kind=='free_object']
        if len(objects)!=1:raise ValueError('此最小任务只支持一个自由物体')
        self.object=objects[0];self.nu=action_map.nu
        self.n_frames=1  # 静态目标只有一个reset入口，不冒充人体动作序列。
        self.action_scale=jnp.asarray(action_map.scale)  # 原actor_step的控制变化日志，源执行器顺序。
        self.jq=np.array([b.source_qpos_start for b in self.layout.blocks if b.kind!='free_object'])
        self.jv=np.array([b.source_qvel_start for b in self.layout.blocks if b.kind!='free_object'])
        self.hist_dim=2*self.nu+15+self.nu  # 手q/v，物体p/R6/v6，上一策略动作
        self.goal_dim=9;self.critic_dim=self.hist_dim+self.goal_dim
        params=np.array([position_scale,orientation_scale,max_distance,action_rate_weight,target_jitter])
        if not np.isfinite(params).all() or np.any(params[:3]<=0) or np.any(params[3:]<0):raise ValueError('任务尺度需正，权重及抖动需非负')
        self.position_scale,self.orientation_scale,self.max_distance,self.action_rate_weight,self.target_jitter=map(float,params)
        q=np.asarray(reset_qpos,dtype=float);v=np.asarray(reset_qvel,dtype=float)
        self.layout.pack(q,v)  # 沿用物理布局校验，不猜测物体地址。
        self.reset_q=jnp.asarray(q);self.reset_v=jnp.asarray(v)
        p,r=self._targets(target_pos,target_quat)
        if p.shape!=(3,) or r.shape!=(4,):raise ValueError('reset目标必须为单个位姿')
        self.target_pos=jnp.asarray(p);self.target_quat=jnp.asarray(r)

    @staticmethod
    def _targets(p,q):
        p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
        if p.shape[-1:]!=(3,) or q.shape!=p.shape[:-1]+(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q,axis=-1),1,rtol=0,atol=1e-10):raise ValueError('目标需有限位置及单位wxyz四元数')
        return p,q

    def initial_state(self,qpos,qvel,*,warm=None,target_pos=None,target_quat=None):
        s=super().initial_state(qpos,qvel,warm=warm);B=s.frame.shape[0]
        p,q=self._targets(self.target_pos if target_pos is None else target_pos,self.target_quat if target_quat is None else target_quat)
        p=np.broadcast_to(p,(B,3));q=np.broadcast_to(q,(B,4))
        return PoseTrackingState(*s,jnp.asarray(p),jnp.asarray(q),jax.random.split(jax.random.PRNGKey(0),B))

    def reset(self,key,frame):
        # frame仅用于兼容调用签名；计数从零开始，不查询人体参考动作。
        p=self.target_pos+jax.random.uniform(key,(3,),dtype=jnp.float64,minval=-self.target_jitter,maxval=self.target_jitter)
        return PoseTrackingState(NativeHandData(self.reset_q,self.reset_v,jnp.zeros_like(self.reset_v)),
            jnp.int32(0),jnp.int32(0),jnp.zeros(self.nu,jnp.float64),FixedHandContext(jnp.float64(1),jnp.zeros(3)),p,self.target_quat,jax.random.key_data(key))

    def pose(self,state):
        i=self.object.source_qpos_start
        return state.dx.qpos[i:i+3],state.dx.qpos[i+3:i+7]

    def policy_obs(self,state,key):
        p,q=self.pose(state);i=self.object.source_qvel_start
        hist=jnp.concatenate((state.dx.qpos[self.jq],state.dx.qvel[self.jv],p,rotation6(q),state.dx.qvel[i:i+6],state.prev_a))
        # 位移误差 + 目标世界朝向；当前朝向已经在hist中。
        return hist,jnp.concatenate((state.target_pos-p,rotation6(state.target_quat)))

    def critic_obs(self,state):
        h,g=self.policy_obs(state,None);return jnp.concatenate((h,g))

    def errors(self,state):
        p,q=self.pose(state)
        return jnp.sum((p-state.target_pos)**2),jnp.maximum(0.,1-jnp.dot(q,state.target_quat)**2)

    def reward(self,state,action,prev_action,prev=None):
        p,r=self.errors(state)
        terms=dict(position=jnp.exp(-p/self.position_scale**2),orientation=jnp.exp(-r/self.orientation_scale**2),
                   action_rate=-self.action_rate_weight*jnp.sum((action-prev_action)**2))
        return self.dt*sum(terms.values()),terms

    def terminated(self,state):
        p,_=self.errors(state)
        finite=jnp.all(jnp.isfinite(self.critic_obs(state))) & jnp.all(jnp.isfinite(state.dx.qpos)) & jnp.all(jnp.isfinite(state.dx.qvel))
        return (~finite)|(p>self.max_distance**2)

    def clip_ended(self,state):return jnp.bool_(False)

    def hazard_features(self,state):
        raise NotImplementedError('不复用G1高度/倾角终止惩罚；本任务使用自身距离终止规则')


    @staticmethod
    def compact(state):
        """保留全部重启所需数据，包括逐episode目标和warm；支持批量。

        手状态没有需重算的运动学缓存，不沿用G1的d0或人体hist。
        """
        return state._asdict()

    @staticmethod
    def restore(compact):
        """供原actor_step用vmap从重放样本恢复单环境状态。"""
        return PoseTrackingState(**compact)
