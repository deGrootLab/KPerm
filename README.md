# KPerm - Toolkit for Analysis of Permeation Cycles in Potassium Channels
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
