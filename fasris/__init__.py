# 让外部方便导入
from .geometry import Geometry, RISConfig, ChannelParams, SystemParams, OptParams, default_problem
from .ris import ris_element_positions, ris_phase_codebook
from .channels import (db2lin, lin2db, friis_pathloss, power_pathloss,
                       rician_scalar, user_to_ris_vector_los, ris_to_fa_vector_rayleigh)
from .objective import (effective_scalar_channel, sinr_and_rate, objective_min_rate)
