"""显式float32/mixed64配置；统一CUDA类型适配，不改变力学公式。"""
import ctypes
import re
import numpy as np


class Numerics:
    def __init__(self,mode='float32'):
        if mode not in ('float32','mixed64'):raise ValueError('numerics必须为float32或mixed64')
        self.mode=mode;self.dtype=np.float32 if mode=='float32' else np.float64
        self.scalar=ctypes.c_float if mode=='float32' else ctypes.c_double
        self.pointer=ctypes.POINTER(self.scalar)

    def finite(self,value,name):
        with np.errstate(over='ignore',invalid='ignore'):result=np.ascontiguousarray(value,dtype=self.dtype)
        if not np.isfinite(result).all():raise ValueError(f'{name}超出{self.dtype.__name__}有限范围')
        return result

    def cuda(self,code):
        if self.mode=='float32':return code
        # NV=33..35的原Cholesky有float32专用PTX倒平方根，不能仅改C类型。
        # 双精度模式使用同一倒平方根的double表达式，保留分解/回代顺序。
        code=code.replace('asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));',
                          'r = 1.0 / sqrt(x);')
        code=re.sub(r'\bfloat\b','double',code)
        for a,b in {'__fmul_rn':'__dmul_rn','__fadd_rn':'__dadd_rn','__fsub_rn':'__dsub_rn','__fmaf_rn':'__fma_rn',
                    'sqrtf':'sqrt','rsqrtf':'rsqrt','sincosf':'sincos','sinf':'sin','cosf':'cos','fabsf':'fabs',
                    'fmaxf':'fmax','fminf':'fmin','powf':'pow','fmaf':'fma'}.items():
            code=re.sub(r'\b'+a+r'\b',b,code)
        return re.sub(r'(?<![\w.])(\d[\d.eE+-]*)f\b',r'\1',code)
