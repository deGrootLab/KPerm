# KPerm - Toolkit for Analysis of Permeation Cycles in Potassium Channels

[![License](https://img.shields.io/github/license/deGrootLab/KPerm)](https://www.gnu.org/licenses/gpl-3.0.en.html) [![PyPI](https://img.shields.io/pypi/v/kperm?color=green)](https://pypi.org/project/kperm/) [![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.3c00061-purple)](https://dx.doi.org/10.1021/acs.jctc.3c00061)

KPerm, released together with our [JCTC](https://dx.doi.org/10.1021/acs.jctc.3c00061) paper, is a Python package specifically for identifying selecivity filter occupancy in molecular dynamics (MD) simulations of potassium channels.

## Installation

You are recommended to install the latest release of KPerm via `pip` in a virtual environment with Python >=3.8.

```bash
pip install kperm
```

## Examples

### Jupyter Notebook

- [Permeation Cycle in MthK with charge-scaling](./docs/notebooks/charge-scaling.ipynb)

### Command-line Interface
```bash
# same as Channel.run(), computing SF occuapncy and identifying permeation events
kperm run -s coord_file -f traj_1 traj_2 traj_3 ...

# speed things up if you are not interested in SF oxygen flip and water occupancy
kperm run -s coord_file -f traj_1 traj_2 traj_3 ... --noFlip --noW

# compute summary of permeation events of all selected simulations 
# use it after running "kperm run"
kperm stats -s coord_file -f traj_1 traj_2 traj_3 ...
```

## Documentation

https://degrootlab.github.io/KPerm/

## Citation

- Lam, C. K., & de Groot, B. L. (2023). Ion Conduction Mechanisms in Potassium Channels Revealed by Permeation Cycles. Journal of Chemical Theory and Computation.doi:[10.1021/acs.jctc.3c00061](https://dx.doi.org/10.1021/acs.jctc.3c00061)
