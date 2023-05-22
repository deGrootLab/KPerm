# KPerm - Toolkit for Analysis of Permeation Cycles in Potassium Channels


[![License](https://img.shields.io/github/license/deGrootLab/KPerm)](https://www.gnu.org/licenses/gpl-3.0.en.html) [![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.3c00061-purple)](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00061)

This package allows you to identify permeation cycles in potassium channels from molecular dynamics (MD) simulation trajectories.

## Conda
You are recommended to install KPerm in a conda environment. If you have installed [Anaconda](https://www.anaconda.com/), create a new environment by running:

```bash
conda create -n kperm python=3.11
```

Once the environment is set up, run:
```bash
conda activate kperm
```
## Installation
Tested with Python 3.11, MDAnalysis 2.4.2, and Numpy 1.24.1.
```bash
git clone https://github.com/tomcklam/KPerm
cd KPerm
pip install .
# you may need to add the kernel
# ipython kernel install --name "kperm" --user
```

You can now try our tutorial (tutorials/charge-scaling/charge-scaling.ipynb).
