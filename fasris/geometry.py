"""
存放：场景几何、RIS 配置、信道与系统参数、优化器参数，以及一个默认问题构造器。
把“输入/常量”集中管理，基线之间可复用。
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Geometry:
    """
    平面 2D 几何（便于直观与快速试验；需要 3D 时可自行扩展为 (x,y,z)）
    - u_des, u_int1, u_int2: 三个发射用户（期望 + 2 干扰）的坐标（米）
    - ris_center: RIS 阵列中心位置（米）
    - fa_box_min / fa_box_max: FAS 可行域的二维边界框（投影约束）
    """
    u_des: np.ndarray
    u_int1: np.ndarray
    u_int2: np.ndarray
    ris_center: np.ndarray
    fa_box_min: np.ndarray
    fa_box_max: np.ndarray

@dataclass
class RISConfig:
    """
    RIS 基本配置：
    - M: 单元数（默认 8×8=64）
    - elem_spacing: 单元间距，以“波长数”计；0.5 λ 常见，抑制栅瓣
    - array_shape: 阵列形状（行、列）
    - phase_bits: 相位量化比特数；2 表示 4 个离散相位
    """
    M: int = 64
    elem_spacing: float = 0.5
    array_shape: tuple = (8, 8)
    phase_bits: int = 2

@dataclass
class ChannelParams:
    """
    信道统计/物理参数：
    - fc_hz: 载频（Hz）
    - K_URIS_dB: 用户→RIS 的 Rician K 因子（dB），越大越近似 LOS
    - K_UFA_dB: 用户→FA 的 Rician K 因子（dB）
    - n_UFA / n_URIS: 幂律路径损耗指数（用于散射项幅度缩放）
    """
    fc_hz: float = 28e9
    K_URIS_dB: float = 5.0
    K_UFA_dB: float = 3.0
    n_UFA: float = 2.2
    n_URIS: float = 2.0

@dataclass
class SystemParams:
    """
    系统功率与噪声：
    - P_des_dBm/P_int_dBm: 期望/干扰发射功率（dBm）
    - N0_dBmHz: 热噪声功率谱密度（dBm/Hz），典型 -174 dBm/Hz
    - B_Hz: 带宽（Hz）
    - noise_figure_dB: 接收机噪声系数（dB），合入等效噪声
    """
    P_des_dBm: float = 20.0
    P_int_dBm: float = 20.0
    N0_dBmHz: float = -174.0
    B_Hz: float = 100e6
    noise_figure_dB: float = 5.0

@dataclass
class OptParams:
    """
    优化器超参：
    - Ns: 每次目标评估的快照数（更鲁棒但更慢）
    - outer_iters: 交替优化的外层迭代数
    - ris_inner_passes: 每次 RIS 更新里，对所有单元做几轮坐标爬山
    - fas_steps: FAS 位置的梯度上升步数
    - fas_step_init: FAS 初始步长（米）
    - fas_fd_eps: 有限差分的扰动（米），控制梯度估计精度/噪声
    - backtrack_beta/c: 回溯线搜索参数（收敛稳定性）
    - rng_seed: 随机种子，保证可复现实验
    """
    Ns: int = 32
    outer_iters: int = 12
    ris_inner_passes: int = 2
    fas_steps: int = 25
    fas_step_init: float = 0.2
    fas_fd_eps: float = 1e-3
    backtrack_beta: float = 0.6
    backtrack_c: float = 1e-3
    rng_seed: int = 20251016

def default_problem():
    """
    构造一个默认几何 + 参数配置，方便快速跑通/做消融。
    你可以直接替换坐标、K 因子、带宽、功率等来贴合你的场景。
    """
    geom = Geometry(
        u_des=np.array([12.0,  5.0]),
        u_int1=np.array([-8.0,  7.0]),
        u_int2=np.array([ 6.0,-10.0]),
        ris_center=np.array([1.0, 0.0]),
        fa_box_min=np.array([-1.0, -1.0]),
        fa_box_max=np.array([ 1.0,  1.0]),
    )
    ris_cfg = RISConfig()
    chp = ChannelParams()
    sysp = SystemParams()
    optp = OptParams()
    return geom, ris_cfg, chp, sysp, optp
