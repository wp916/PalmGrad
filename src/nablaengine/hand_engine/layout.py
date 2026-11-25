"""hand专用索引。只有规定运动手掌、hinge和最多4个free物体，不分派G1布局。"""
from dataclasses import dataclass
import numpy as np
from nablaengine.model.hand import from_mujoco


def _readonly(value,dtype):
    a=np.ascontiguousarray(value,dtype=dtype);a.setflags(write=False);return a


@dataclass(frozen=True)
class HandTopology:
    """kind: 0=world,1=prescribed palm,2=hinge,3=free object。

    meta保留现有hand CUDA入口的四段传输形式：parent/qpos_start/qvel_start/kind。
    world/palm没有动态地址，明确填-1。children为CSR，无每体3子节点限制。
    body_dof_mask直接按动态自由度编号置位，不借助body编号移位猜测自由度。
    """
    parent: np.ndarray
    qpos_start: np.ndarray
    qvel_start: np.ndarray
    kind: np.ndarray
    dof_count: np.ndarray
    root: np.ndarray
    child_offsets: np.ndarray
    children: np.ndarray
    support_offsets: np.ndarray
    support_dofs: np.ndarray
    body_dof_mask: np.ndarray
    roots: np.ndarray
    axes: np.ndarray
    depth: np.ndarray
    meta: np.ndarray

    def contact_mask(self,body1,body2):
        for b in (body1,body2):
            if not isinstance(b,(int,np.integer)) or b<0 or b>=len(self.parent):
                raise ValueError('body id超出hand模型范围')
        return int(self.body_dof_mask[body1]) ^ int(self.body_dof_mask[body2])


def build_topology(model,layout):
    audited=from_mujoco(model,palm=layout.palm_name,
                       objects=[b.name for b in layout.blocks if b.kind=='free_object'],
                       joints=[b.name for b in layout.blocks if b.kind=='hinge'])
    if audited!=layout:raise ValueError('layout与源模型不一致')
    if model.nbody>64 or layout.nv>64:
        raise ValueError('当前hand索引方案最多64刚体（含world）、64动态自由度，需显式扩展位宽')
    nb=int(model.nbody);parent=np.array(model.body_parentid,dtype=np.int32)
    if any(parent[b]>=b for b in range(1,nb)):raise ValueError('父节点必须先于子节点')
    qadr=np.full(nb,-1,np.int32);dadr=qadr.copy();kind=np.zeros(nb,np.int32);count=kind.copy()
    axes=np.zeros((nb,3));root=np.zeros(nb,np.int32);depth=root.copy()
    kind[layout.palm_body_id]=1
    for block in layout.blocks:
        b=block.body_id;qadr[b]=block.qpos_start;dadr[b]=block.qvel_start;count[b]=block.nv
        kind[b]=2 if block.kind=='hinge' else 3
        if block.kind=='hinge':axes[b]=model.jnt_axis[block.joint_id]
    child_lists=[[] for _ in range(nb)]
    support_lists=[[] for _ in range(nb)];masks=[]
    for b in range(1,nb):
        p=int(parent[b]);child_lists[p].append(b);depth[b]=depth[p]+1
        root[b]=b if p==0 else root[p]
        support_lists[b]=sorted(support_lists[p]+list(range(dadr[b],dadr[b]+count[b])))
    for dofs in support_lists:masks.append(sum(1<<d for d in dofs))
    def csr(rows):
        offsets=np.r_[0,np.cumsum([len(r) for r in rows])]
        data=[item for row in rows for item in row]
        return _readonly(offsets,np.int32),_readonly(data,np.int32)
    co,children=csr(child_lists);so,support=csr(support_lists)
    roots=[layout.palm_body_id]+[b.body_id for b in layout.blocks if b.kind=='free_object']
    meta=np.concatenate([parent,qadr,dadr,kind])
    return HandTopology(*[_readonly(a,np.int32) for a in (parent,qadr,dadr,kind,count,root)],
                        co,children,so,support,_readonly(masks,np.uint64),_readonly(roots,np.int32),
                        _readonly(axes,np.float64),_readonly(depth,np.int32),_readonly(meta,np.int32))
