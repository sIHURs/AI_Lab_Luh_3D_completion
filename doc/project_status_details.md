# Problem Log

## Pytorch3D install issue (26.05)

> When setup the environment for the paper "Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion" ([code](https://github.com/PRBonn/LiDiff)), I had issues with installing the PyTorch3D package on Windows. The installation process is quite cumbersome. I have now matched the versions of CUDA (12.1), Torch (2.2.0), and NVIDIA CUB (2.0.1), and downloaded the corresponding version of PyTorch3D (0.7.6) as a compressed package. However, it seems that on Windows, PyTorch3D needs to be compiled and installed using the x64 Native Tools Command Prompt for VS 2019 with administrative privileges?

**Possible solutions:**

In this project the only function used is the _chamfer_distance_ from pytorch3D, which is used only during the refinement network training to compute the chamfer distance loss.

Can use a alternativ [package](https://pypi.org/project/chamferdist/) for computing the chamfer distance or compute a new one to fit the project, cuz the package is a different implementation of the chamfer distance, this my lead to different results than the ones reported in the paper.

## ...

## MinkowskiEngine / PytorchEMD(DiT-3D) Install issue :( (28.05)

MinkowskiEngine only supported in Ubuntu, there is a possible solution on windows but its cpu only. On colabs it also cant work.

**Possible solutions:**
try on l3s server only

## rl.exe Error, C++ Compiler

```bash
subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1
```

In DiT-3D repo modules dictionary need c++ compiler. But the system can's find rl.exe.

`can't solved, try to use l3s server, and focus on theory and coding implementaion in juni (05.06)`

- [ ] diffusion model
- [ ] transformer
- [ ] ViT, Vision Transformer
- [ ] DiT, Diffusion Transformer

## Learning diffuion model coding from stable diffusion (05.06)

stable diffusion, is a text-to-image deep learning model, based on diffusion model, and its a latent diffusion model, in which we don't learn the distribution p(x) of our data set of images, but rather, the distribution of a latent representation of our data by using a Variational Autoencoder (VAE).

Programming content:

- DDPM/ Sampler
- diffuison process
- VAE
- Classifier-Free Guidence
- Self-Attention/Cross-Attention
- CLIP
- U-Net
- Implementaion of model loader and how to make a pipeline and how to make a training/inference demo.

Source code from [pytorch-stable-diffuison](https://github.com/kjsman/stable-diffusion-pytorch?tab=readme-ov-file)

## Dive deep into [LiDiff](https://github.com/PRBonn/LiDiff?tab=readme-ov-file) (Next)

Find out how the author achieved in code, from shape reconstruction to scene reconstruction.
