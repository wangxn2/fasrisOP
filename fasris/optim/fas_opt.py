"""
FAS 位置优化（连续二维）：
- 使用投影梯度上升（Projected Gradient Ascent）：
  * 有限差分估计梯度（中心差分更稳健）
  * 回溯线搜索（Armijo 条件）确保每步都有提升
  * 投影到可行域的盒子约束（防止越界）
- 简洁可靠，便于替换为更强的算法（PSO/SA/贝叶斯优化等）
"""
import numpy as np
from numpy.random import default_rng
from ..objective import objective_min_rate

def project_to_box(p, lo, hi):
    """把点 p 投影到 [lo, hi] 的盒状可行域内。"""
    return np.minimum(np.maximum(p, lo), hi)

def finite_diff_grad(f, x, eps):
    """
    中心差分梯度，x ∈ R^2。
    eps 过大 ⇒ 偏差大；过小 ⇒ 数值噪声/随机性放大。1e-3 米通常足够稳定。
    """
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i]=1.0
        g[i] = (f(x+eps*e) - f(x-eps*e))/(2*eps)
    return g

def optimize_fas_position(des_xy, int1_xy, int2_xy, fa_xy_init, ris_pos, phi,
                          chp, sysp, optp, rng_master, box_lo, box_hi):
    # 初始位置
    x = fa_xy_init.copy()
    # 固定本地 RNG，避免每次梯度评估“快照不同”造成噪声
    rng_local = default_rng(rng_master.integers(0, 2**31-1))

    # 目标函数闭包：只对 x（FAS 位置）敏感
    f = lambda xc: objective_min_rate(des_xy, int1_xy, int2_xy, xc, ris_pos, phi,
                                      chp, sysp, rng_local, optp.Ns)
    f_cur = f(x)

    step = optp.fas_step_init
    for _ in range(optp.fas_steps):
        # 估计梯度
        g = finite_diff_grad(f, x, optp.fas_fd_eps)
        if np.linalg.norm(g) < 1e-10:  # 梯度很小，认为收敛
            break

        # 回溯线搜索（找到能显著提升 f 的步长）
        step_try = step
        improved = False
        while step_try > 1e-6:
            x_new = project_to_box(x + step_try*g, box_lo, box_hi)
            f_new = f(x_new)
            # Armijo 条件（上升版）：f(x_new) ≥ f(x) + c*α*||g||^2
            if f_new >= f_cur + optp.backtrack_c*step_try*(np.linalg.norm(g)**2):
                x, f_cur, step = x_new, f_new, step_try/optp.backtrack_beta  # 适度放宽下一轮初始步长
                improved = True
                break
            step_try *= optp.backtrack_beta  # 缩小步长再试
        if not improved:
            break
    return x, f_cur
