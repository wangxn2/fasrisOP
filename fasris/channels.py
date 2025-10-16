"""
信道建模（改版）：
- 用户→RIS：纯 LOS（相位=几何，幅度=Friis）
- RIS→FA：瑞利衰落（零均值 CN，方差由 Friis 幅度决定 ⇒ 位置相关）
- 用户→FA：保持 Rician（如需改为 Rayleigh，可按同样方式把 LOS 权重设为 0）
"""
import numpy as np

def db2lin(x_db: float) -> float:
    return 10.0**(x_db/10.0)

def lin2db(x: float) -> float:
    return 10.0*np.log10(np.maximum(x, 1e-30))

def friis_pathloss(dist: np.ndarray, fc_hz: float, Gt=1.0, Gr=1.0) -> np.ndarray:
    """
    Friis 幅度：|h| = (λ / (4πd)) * sqrt(Gt*Gr)
    注意：这是“复信道幅度”的标度，不是功率；功率会是它的平方。
    """
    c = 3e8
    lam = c / fc_hz
    return Gt*Gr*(lam/(4*np.pi*np.maximum(dist,1e-6)))

def power_pathloss(dist: np.ndarray, n: float, ref_d=1.0) -> np.ndarray:
    """
    幂律幅度：功率 ~(d/ref)^(-n) ⇒ 幅度 ~(d/ref)^(-n/2)
    在本改版里主要用于用户→FA 的散射项幅度（Rician 的 diffuse）。
    """
    return (np.maximum(dist,1e-6)/ref_d)**(-n/2.0)

# ---------------------------- 用户→FA（保持 Rician） ----------------------------

def rician_scalar(from_xy: np.ndarray, to_xy: np.ndarray, fc_hz: float,
                  K_dB: float, n_exp: float, rng: np.random.Generator):
    """
    标量 Rician：h = a*LOS + b*Diffuse
    - LOS：单位幅度（把距离衰减交给散射项或另外的常数；也可以把 Friis 幅度并入 LOS，二者是一致的缩放）
    - Diffuse：CN(0, σ^2)，σ 由幂律幅度给出
    """
    c = 3e8
    lam = c / fc_hz
    d = np.linalg.norm(to_xy - from_xy)
    h_los = np.exp(-1j*2*np.pi*d/lam)                   # 单位幅度的 LOS 相位
    mag_diff = power_pathloss(np.array([d]), n=n_exp)[0] # 散射幅度（随距离衰减）
    K = db2lin(K_dB)
    a = np.sqrt(K/(K+1)); b = np.sqrt(1/(K+1))
    diffuse = (rng.normal(scale=mag_diff/np.sqrt(2)) + 1j*rng.normal(scale=mag_diff/np.sqrt(2)))
    return a*h_los + b*diffuse

# ---------------------------- 用户→RIS（改为纯 LOS 向量） ----------------------------

def user_to_ris_vector_los(user_xy: np.ndarray, ris_pos: np.ndarray, fc_hz: float):
    """
    用户→RIS：纯 LOS，逐单元相干：
    h_ur[m] = exp(-j 2π d_m / λ) * Friis(d_m)
    这样 RIS 端看到的是“方向一致的相位前”（阵面上准平面波），
    并且距离衰减真实地反映了用户到每个单元的几何差异。
    """
    c = 3e8
    lam = c / fc_hz
    dvec = ris_pos - user_xy[None,:]     # (M,2)
    dist = np.linalg.norm(dvec, axis=-1) # (M,)
    phase = np.exp(-1j*2*np.pi*dist/lam) # 相位项
    mag   = friis_pathloss(dist, fc_hz)  # 幅度项（可改为常数以弱化幅度差异）
    return phase * mag                   # (M,)

# ---------------------------- RIS→FA（改为瑞利向量） ----------------------------

def ris_to_fa_vector_rayleigh(ris_pos: np.ndarray, fa_xy: np.ndarray,
                              fc_hz: float, rng: np.random.Generator):
    """
    RIS→FA：Rayleigh（零均值 CN），逐单元独立（需要相关性时可自行加入相关建模）
    g[m] ~ CN(0, σ_m^2)，其中 σ_m = Friis(d_m)
    —— 关键点：σ_m 取决于“元素 m 到 FA 的距离”，
       因此 FAS 的位置会改变 {σ_m}，进而改变 RIS 叠加的统计结果。
    """
    dvec = fa_xy[None,:] - ris_pos      # (M,2)
    dist = np.linalg.norm(dvec, axis=-1)
    sigma = friis_pathloss(dist, fc_hz) # 把 Friis 幅度当作瑞利的标准差标度
    # 复高斯：每个元素独立 CN(0, sigma^2)
    g = (rng.normal(size=dist.shape, scale=sigma/np.sqrt(2))
         + 1j*rng.normal(size=dist.shape, scale=sigma/np.sqrt(2)))
    return g  # (M,)
