# Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion

They use a Unet-based diffusion model to complete the sparse LiDAR input, leverage the diffusion process as a point-wise local problem. During the diffusion process, they disentangle the scene data distribution, learning only the point's local neighborhood distribution. Through their formulation, they can directly operate on the 3D points, obtaining a complete scene representation from a single LiDAR scan. The LiDAR data completed by the diffusion model is then refined by a refinement model, significantly enhancing the reconstruction of the scene., [source code](https://github.com/PRBonn/LiDiff?tab=readme-ov-file)

## Env
Code is tested on Ubuntu 24.04, cuda 11.8, pytorch 2.1.2, python 3.10, gcc 11.4.

### Minkowski Engine
The model is implemented by Minkowski Engine.
To install Minkowski Engine localy with cuda 11.X and anaconda env:
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
more information find [here](https://github.com/NVIDIA/MinkowskiEngine#anaconda)

### pytroch3D
Install [pytroch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) with anaconda env.
```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

#if Cuda older than 11.7
#conda install -c bottler nvidiacub 
# Demos and examples
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

# Tests/Linting
pip install black usort flake8 flake8-bugbear flake8-comprehensions

# install from a local clone
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```