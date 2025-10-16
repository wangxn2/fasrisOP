"""
简单基线：随机 RIS 相位 + 随机 FAS 位置，多次尝试取最优（min-rate 最大）。
用于 sanity check 与上界/下界比较。
"""
import numpy as np
from numpy.random import default_rng
from fasris.ris import ris_element_positions, ris_phase_codebook
from fasris.objective import objective_min_rate

def random_baseline(geom, ris_cfg, chp, sysp, Ns=32, trials=200, seed=1):
    rng = default_rng(seed)
    c=3e8; lam=c/chp.fc_hz
    ris_pos = ris_element_positions(ris_cfg, lam, geom.ris_center)
    codebk = ris_phase_codebook(bits=ris_cfg.phase_bits)

    best = (-np.inf, None, None)
    for _ in range(trials):
        # 随机采样 RIS 相位与 FAS 位置
        phi = rng.choice(codebk, size=ris_cfg.M)
        fa = rng.uniform(geom.fa_box_min, geom.fa_box_max)
        # 评估当前组合的 min-rate
        val = objective_min_rate(geom.u_des, geom.u_int1, geom.u_int2, fa, ris_pos, phi,
                                 chp, sysp, rng, Ns)
        if val > best[0]:
            best = (val, phi, fa)
    return {"min_rate_bps": float(best[0]), "phi": best[1], "fa_xy": best[2]}
