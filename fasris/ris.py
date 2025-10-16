"""
RIS 几何与相位码本工具：
- 计算阵列元素在平面中的坐标（用于路径长度与相位）
- 构造离散相位码本（2^b 个相位）
"""
import numpy as np

def ris_phase_codebook(bits: int) -> np.ndarray:
    """
    生成均匀量化的相位码本（弧度）。
    例：bits=2 => [0, π/2, π, 3π/2]
    """
    L = 2 ** bits
    return np.arange(L) * (2 * np.pi / L)

def ris_element_positions(ris_cfg, wavelength: float, ris_center_xy: np.ndarray) -> np.ndarray:
    """
    返回 (M,2) 的平面坐标，阵列在 XY 平面内以中心对称分布。
    - elem_spacing 以“波长”为单位，这样即使改载频，阵列实际米制间距也会自动更新
    """
    rows, cols = ris_cfg.array_shape
    assert rows * cols == ris_cfg.M, "array_shape 与 M 不一致"
    dx = ris_cfg.elem_spacing * wavelength
    dy = ris_cfg.elem_spacing * wavelength
    xs = (np.arange(cols) - (cols - 1)/2) * dx
    ys = (np.arange(rows) - (rows - 1)/2) * dy
    gx, gy = np.meshgrid(xs, ys)
    # 叠加阵列中心位移，得到全局坐标
    return np.stack([gx.ravel(), gy.ravel()], axis=1) + ris_center_xy[None,:]
