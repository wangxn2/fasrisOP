"""
交替优化主循环：
- 给定初始 FAS 位置与 RIS 相位（全 0），反复：
  1) 固定 FAS，更新 RIS 相位（离散坐标爬山）
  2) 固定 RIS，相位，更新 FAS 位置（投影梯度上升）
- 每轮用独立 RNG 评估 min-rate，避免“过拟合某个快照集”
"""
import numpy as np
from numpy.random import default_rng
from ..ris import ris_element_positions, ris_phase_codebook
from ..objective import (objective_min_rate, sinr_and_rate)
from .ris_opt import optimize_ris_phases
from .fas_opt import optimize_fas_position

def run_alt_optimization(geom, ris_cfg, chp, sysp, optp):
    rng_master = default_rng(optp.rng_seed)
    c = 3e8; lam = c / chp.fc_hz

    # 计算 RIS 各单元全局坐标
    ris_pos = ris_element_positions(ris_cfg, lam, geom.ris_center)
    # 初始化：RIS 相位设为 0；FAS 放在可行域中心
    phi = np.zeros(ris_cfg.M)
    fa_xy = 0.5*(geom.fa_box_min + geom.fa_box_max)

    history = []
    best_obj, best_sol = -np.inf, None

    for it in range(optp.outer_iters):
        # === Step 1: RIS 更新 ===
        phi, _ = optimize_ris_phases(geom.u_des, geom.u_int1, geom.u_int2,
                                     fa_xy, ris_pos, phi, chp, sysp, optp, rng_master)
        # === Step 2: FAS 更新 ===
        fa_xy, _ = optimize_fas_position(geom.u_des, geom.u_int1, geom.u_int2,
                                         fa_xy, ris_pos, phi, chp, sysp, optp, rng_master,
                                         geom.fa_box_min, geom.fa_box_max)
        # === 评估（用新的 RNG，避免“测量偏差”）===
        rng_eval = default_rng(rng_master.integers(0, 2**31-1))
        obj_eval = objective_min_rate(geom.u_des, geom.u_int1, geom.u_int2,
                                      fa_xy, ris_pos, phi, chp, sysp, rng_eval, optp.Ns)

        history.append({
            "iter": it+1,
            "min_rate_bps": float(obj_eval),
            "fa_xy": fa_xy.copy(),
        })
        # 记录最佳
        if obj_eval > best_obj:
            best_obj = obj_eval
            best_sol = {"phi": phi.copy(), "fa_xy": fa_xy.copy(), "min_rate_bps": float(obj_eval)}

        print(f"[Iter {it+1:02d}] min-rate = {obj_eval/1e6:.3f} Mb/s, FA=({fa_xy[0]:.2f},{fa_xy[1]:.2f})")

    return best_sol, history, ris_pos
