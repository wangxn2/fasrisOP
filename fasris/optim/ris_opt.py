"""
RIS 相位优化（离散码本）：
- 坐标爬山（coordinate ascent）：逐单元枚举所有量化相位，选择能提升目标的那一个
- 好处：稳定、实现简单；坏处：大规模 M 时耗时，可按块/并行/启发式加速
"""
import numpy as np
from numpy.random import default_rng
from ..ris import ris_phase_codebook
from ..objective import objective_min_rate

def optimize_ris_phases(des_xy, int1_xy, int2_xy, fa_xy, ris_pos, phi_init,
                        chp, sysp, optp, rng_master):
    # 从当前相位出发
    phi = phi_init.copy()
    # 固定码本（2^bits 个相位）
    codebk = ris_phase_codebook(bits=2)
    M = phi.size

    # 为避免 RIS 更新过程中“快照噪声”导致不稳定比较，这里固定本地 RNG
    rng_local = default_rng(rng_master.integers(0, 2**31-1))

    # 先评估当前目标值
    base = objective_min_rate(des_xy, int1_xy, int2_xy, fa_xy, ris_pos, phi,
                              chp, sysp, rng_local, optp.Ns)

    # 做若干次“全阵列扫描”（inner passes）
    for _ in range(optp.ris_inner_passes):
        improved = False
        # 逐单元贪婪更新
        for m in range(M):
            best, best_phi = base, phi[m]
            # 尝试该单元的所有候选相位
            for cand in codebk:
                if np.isclose(cand, phi[m]): 
                    continue
                phi_try = phi.copy()
                phi_try[m] = cand
                val = objective_min_rate(des_xy, int1_xy, int2_xy, fa_xy, ris_pos, phi_try,
                                         chp, sysp, rng_local, optp.Ns)
                if val > best:
                    best, best_phi = val, cand
            # 若找到更好相位，则接受
            if not np.isclose(best_phi, phi[m]):
                phi[m] = best_phi
                base = best
                improved = True
        # 一轮扫描无改进，提前停止
        if not improved: 
            break
    return phi, base
