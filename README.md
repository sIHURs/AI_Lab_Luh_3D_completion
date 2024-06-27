# 2024 AI Lab Shape Completion for Outdoor Environments

## Project Description

### What is scene completion

\
Computer vision techniques are essential components of the perception systems in autonomous vehicles. These methods are used to interpret the vehicle's surroundings based on sensor data. 3D LiDAR sensors are often employed to gather sparse 3D point clouds from the environment. However, unlike human perception, these systems face challenges in inferring the unseen portions of the scene from these sparse point clouds. To address this issue, the scene completion task focuses on predicting the missing parts in the LiDAR measurements to create a more comprehensive representation of the scene.

### Project's goal

\
The main objective of this project is to use diffusion models for scene completion from a single 3D LiDAR scan. This project builds on two studies: in the first study [[1]](#references), they used diffusion models and operated directly on points, reformulating the noise and denoise diffusion processes to enable the diffusion model to work efficiently at the scene scale for shape reconstruction. In the second study [[2]](#references), they proposed an innovative Diffusion Transformer for 3D shape generation, replacing the existing U-Net in diffusion models, which significantly improved the quality of 3D shape generation. **The goal of this project is to replace the diffusion model from the first study with the Transformer model from the second study and achieve scene reconstruction.**

## Project Challenges

- [ ] TODO

## Installation

- [ ] TODO - requirements.txt / setup.py

## Dataset Description

- [ ] TODO

## Examples

- [ ] TODO - traning
- [ ] TODO - ...

## Trained model

- [ ] TODO

## Project status

- [x] Startup: try SemanticKITTI Dataset, and diffuison model training
  - [x] to document(**Done**: Understand and try some programming on the datasets ShapeNet and KITTI360-test-demo. SemanticKITTI (80GB) is too large to download on my own computer. Use Open3D for visualization and PyntCloud for voxelization.) `13.06`
  - [ ] ? voxelization module from DiT3D ( actually from PVCnn)

- [ ] TODO - repoduce the results from the two stduies\
`Device issues can't be resolved, linux system required (PytorchEMD, MinkowskiEngine, pytorch3d)`
  - [code[1]](https://github.com/PRBonn/LiDiff)
  - [code[2]](https://github.com/DiT-3D/DiT-3D)
- [ ] TODO - try replace the model with Diffusion Transformer 3D
  - should convert the point cloud into a voxel representation or figure out other way to direct use the point could data.
  - ...
- [ ] compare the results
  - metrics: $CD$(chamfer distance), $JSD_{BEV}$(jensen-Shannon divergence), $IoU$ (intersection-over-union)
- [ ] ...
## Modules
### pytorchEMD
PyTorch Wrapper for Earth-Mover-Distance (EMD) for 3D point cloud regression, source from [github](https://github.com/daerduoCarey/PyTorchEMD).

But the TH/THC (`fatal error: THC/THC.h: No such file or directory`) include files have been removed in the latest pytorch version. I removed THC/THC.h and updated all the deprecated functions, you can find code in [here](tryout/PyTorchEMD/).

build with command:
```bash
pip install .
```
The code is tested on Ubuntu 24.04, pytroch 2.1.2, CUDA 11.8.


## Recommaned papers
DDPM:
-  [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970) - math theory

Vision Transformer:

- [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)

Diffusion Transformer:

- [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)

PointNet:

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593)

3D Completion:

- [A CONDITIONAL POINT DIFFUSION-REFINEMENT PARADIGM FOR 3D POINT CLOUD COMPLETION](https://arxiv.org/pdf/2112.03530)



## References

1. [Nunes, L., Marcuzzi, R., Mersch, B., Behley, J., & Stachniss, C. (2024). Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion. arXiv preprint arXiv:2403.13470.](https://arxiv.org/abs/2403.13470)

2. [Mo, S., Xie, E., Chu, R., Hong, L., Niessner, M., & Li, Z. (2024). Dit-3d: Exploring plain diffusion transformers for 3d shape generation. Advances in Neural Information Processing Systems, 36.](https://arxiv.org/abs/2307.01831)
