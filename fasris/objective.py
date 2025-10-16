"""
目标计算（改版）：
- 用户→RIS：改为纯 LOS（user_to_ris_vector_los）
- RIS→FA：改为瑞利（ris_to_fa_vector_rayleigh，注意每个快照都要重采样）
- 用户→FA：保持 Rician
"""
import numpy as np
from .channels import (db2lin, rician_scalar,
                       user_to_ris_vector_los, ris_to_fa_vector_rayleigh)

def effective_scalar_channel(user_xy, fa_xy, ris_pos, phi, chp, rng):
    """
    h_eff = h_d (Rician) + g^T diag(e^{jφ}) h_ur
      - h_ur：用户→RIS（LOS 向量，确定性给定几何）
      - g   ：RIS→FA（Rayleigh 向量，每个快照重采样，位置改变其方差谱）
      - h_d ：用户→FA（Rician，保持原设）
    """
    # 直射（用户→FA）：Rician
    h_d  = rician_scalar(user_xy, fa_xy, chp.fc_hz, chp.K_UFA_dB, chp.n_UFA, rng)
    # 用户→RIS：LOS（不依赖 rng）
    h_ur = user_to_ris_vector_los(user_xy, ris_pos, chp.fc_hz)
    # RIS→FA：Rayleigh（依赖 rng，每个快照独立）
    g    = ris_to_fa_vector_rayleigh(ris_pos, fa_xy, chp.fc_hz, rng)
    # 反射叠加
    return h_d + np.sum(g * np.exp(1j*phi) * h_ur)

def sinr_and_rate(des_xy, int1_xy, int2_xy, fa_xy, ris_pos, phi, chp, sysp, rng, Ns):
    """
    与原版相同，但注意 effective_scalar_channel 里现在含有 Rayleigh 的 g，
    因此每个快照都会重采样 RIS→FA，小尺度起伏对 FAS 位置更敏感。
    """
    P_des = db2lin(sysp.P_des_dBm - 30.0)
    P_int = db2lin(sysp.P_int_dBm - 30.0)
    N0    = db2lin(sysp.N0_dBmHz - 30.0)
    N0_eff = N0 * db2lin(sysp.noise_figure_dB)
    sigma2 = N0_eff * sysp.B_Hz

    rates = np.zeros(Ns)
    for i in range(Ns):
        h_des = effective_scalar_channel(des_xy, fa_xy, ris_pos, phi, chp, rng)
        h_i1  = effective_scalar_channel(int1_xy, fa_xy, ris_pos, phi, chp, rng)
        h_i2  = effective_scalar_channel(int2_xy, fa_xy, ris_pos, phi, chp, rng)

        num = P_des * (np.abs(h_des)**2)
        den = P_int * (np.abs(h_i1)**2 + np.abs(h_i2)**2) + sigma2
        sinr = num / np.maximum(den, 1e-30)
        rates[i] = sysp.B_Hz * np.log2(1 + sinr)
    return rates

def objective_min_rate(des_xy, int1_xy, int2_xy, fa_xy, ris_pos, phi, chp, sysp, rng, Ns):
    return np.min(sinr_and_rate(des_xy, int1_xy, int2_xy, fa_xy, ris_pos, phi, chp, sysp, rng, Ns))
