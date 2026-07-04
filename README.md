# DCx
Implementation of "Dual Contouring over Expanded Cubes (DCx) for Zero-Level Set Extraction from Neural Unsigned Distance Functions" (SIGGRAPH 2026))



## env
'''
conda create -n dcx python=3.9 -y
conda activate dcx
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8. cuda-cccl=11.8. -y
pip install open3d scikit-image tqdm pyhocon==0.3.57 trimesh PyMCubes scipy point_cloud_utils==0.29.7

'''
### NUDF
We use CAPUDF to compute NUDF
'''
cd CAPUDF/extensions/chamfer_dist
python setup.py install
'''
We appreciate their work CAPUDF(https://github.com/junshengzhou/CAP-UDF)

### GTUDF
We use cubvh to compute GTUDF
'''
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
'''

