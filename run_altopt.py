"""
程序入口（示例）：
- 构造默认问题
- 运行交替优化
- 打印结果与 RIS 量化索引（便于下发到硬件/仿真）
"""
from fasris.geometry import default_problem
from fasris.optim.altopt import run_alt_optimization
from fasris.ris import ris_phase_codebook
import numpy as np

def main():
    geom, ris_cfg, chp, sysp, optp = default_problem()
    best, hist, ris_pos = run_alt_optimization(geom, ris_cfg, chp, sysp, optp)

    print("\n=== Optimization Result ===")
    print(f"Best min-rate: {best['min_rate_bps']/1e6:.3f} Mb/s")
    print(f"Best FAS position: x={best['fa_xy'][0]:.3f}, y={best['fa_xy'][1]:.3f}")

    # 将相位映射为码本索引（0..L-1），方便记录/下发
    codebk = ris_phase_codebook(bits=ris_cfg.phase_bits)
    idx = ((best["phi"][:,None] - codebk[None,:])**2).argmin(axis=1)
    print(f"RIS phase indices (row-major {ris_cfg.array_shape}):")
    print(idx.reshape(ris_cfg.array_shape))

if __name__ == "__main__":
    main()
