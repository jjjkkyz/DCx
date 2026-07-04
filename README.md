<div align="center">
  <h1>🧊 DCx</h1>
  <p>
    <b>Dual Contouring over Expanded Cubes (DCx) for Zero-Level Set Extraction from Neural Unsigned Distance Functions</b><br>
    <i>(ACM SIGGRAPH 2026)</i>
  </p>

  <p>
    <a href="https://github.com/jjjkkyz/DCx/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9-green.svg">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.2.1-ee4c2c.svg">
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8-76B900.svg">
  </p>
</div>

---

## 📝 About

This repository contains the official implementation of the SIGGRAPH 2026 paper: **"Dual Contouring over Expanded Cubes (DCx) for Zero-Level Set Extraction from Neural Unsigned Distance Functions"**.

## 🛠️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage your environment. Follow the steps below to set up the dependencies:

```bash
# 1. Create and activate a new conda environment
conda create -n dcx python=3.9 -y
conda activate dcx

# 2. Install PyTorch with CUDA 11.8
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install CUDA toolkit and related packages for CAPUDF
conda install -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8 cuda-cccl=11.8 -y

# 4. Install other Python dependencies
pip install open3d scikit-image tqdm pyhocon==0.3.57 trimesh PyMCubes scipy point_cloud_utils==0.29.7

'''
We use CAPUDF to compute Neural Unsigned Distance Functions (NUDF). You need to compile the Chamfer Distance extension first:

'''bash
cd CAPUDF/extensions/chamfer_dist
python setup.py install
'''
We use cubvh to compute Ground Truth Unsigned Distance Functions (GTUDF). Install it directly via git:
'''bash
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
'''



