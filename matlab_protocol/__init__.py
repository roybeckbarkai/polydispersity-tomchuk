"""MATLAB-faithful TENOR-SAXS protocol port.

This package mirrors the workflow in `matlab-new/`:

- initialize instrument / simulation / ensemble parameters
- generate 2D SAXS detector images with the MATLAB protocol
- run TENOR landscape analysis
- save MATLAB-style HDF5 outputs
- reproduce the batch violin plot
"""

from .params import EnsembleParams, InstrumentParams, SimulationParams, init_tenor_params
from .simulation import scatter2d
from .tenor_analysis import tenor_protocol_4_26

__all__ = [
    "EnsembleParams",
    "InstrumentParams",
    "SimulationParams",
    "init_tenor_params",
    "scatter2d",
    "tenor_protocol_4_26",
]
