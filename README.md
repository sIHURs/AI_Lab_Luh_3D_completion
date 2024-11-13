# 2024 AI Lab Shape Completion for Outdoor Environments

## Project Description

### What is 3D point cloud completion

\
With the widespread adoption of 3D scanning devices such as LiDAR, laser scanners, and RGB-D cameras, point clouds have become easier to capture, sparking significant research in fields like robotics, autonomous driving, 3D modeling, and manufacturing. However, due to challenges such as occlusion, reflection, transparency, and limitations in device resolution and angles, the raw point clouds directly captured by these devices are often sparse and partial. Therefore, generating complete point clouds from partial observations is crucial for advancing downstream applications. For example, 3D object detection tasks rely on complete point clouds to maintain state-of-the-art (SOTA) performance. Distant cars captured by LiDAR are often sparse and difficult to detect. Recent advances in deep learning (DL) have significantly improved the capability and robustness of point cloud completion, and the emergence of generative models has introduced new possibilities for this task.
<!-- Computer vision techniques are essential components of the perception systems in autonomous vehicles. These methods are used to interpret the vehicle's surroundings based on sensor data. 3D LiDAR sensors are often employed to gather sparse 3D point clouds from the environment. However, unlike human perception, these systems face challenges in inferring the unseen portions of the scene from these sparse point clouds. To address this issue, the scene completion task focuses on predicting the missing parts in the LiDAR measurements to create a more comprehensive representation of the scene. -->

### Project's goal

\
The main objective of this project is to use diffusion models combined with Vision Transformers to complete partial object point clouds. This project is primarily based on the research in [[2]DiT-3D](#references), where they successfully applied the Diffusion Transformer to the task of 3D point cloud generation. 

This project will further explore the potential of this model for point cloud completion by introducing a new conditional encoder for partial point clouds and incorporating AdaptFormer to improve the efficiency of model fine-tuning.

*(Clarify: The original objective was to complete scene point clouds based on the work in paper [[1]LiDiff](#references). The goal was to explore the potential of using a Transformer, instead of the U-Net used in the diffusion model of that paper, for scene point cloud completion within the same framework. However, due to the large size of scene point cloud data, which demands very high GPU and CPU RAM capacity, the hardware performance was insufficient to meet the requirements. Therefore, the focus was shifted to the aforementioned goal.)*

<figure>
    <img src="assets/example0.png" alt="This is an image">
    <figcaption>Figure 1: the left is the partial point cloud, the middle is the ground truth, and the right is the point cloud reconstructed by the model. This pichture is renderd by mitsuba</figcaption>
</figure>

<!-- (The main objective of this project is to use diffusion models for scene completion from a single 3D LiDAR scan. This project builds on two studies: in the first study [[1]](#references), they used diffusion models and operated directly on points, reformulating the noise and denoise diffusion processes to enable the diffusion model to work efficiently at the scene scale for shape reconstruction. In the second study [[2]](#references), they proposed an innovative Diffusion Transformer for 3D shape generation, replacing the existing U-Net in diffusion models, which significantly improved the quality of 3D shape generation. **The goal of this project is to replace the diffusion model from the first study with the Transformer model from the second study and achieve scene reconstruction.**) -->

<!-- 
## Project Challenges

- [ ] TODO -->

## Installation

Install the following environments:
```
python==3.10
pytorch==2.1.2
torchvision==0.17.1
cudatoolkit==11.8
matplotlib==3.9.0
tqdm==4.66.4
open3d==0.18.0
trimesh=3.7.12
scipy==1.14.1
```
Build Cuda extension moduls: \
Tested on ubuntu 24.04, cuda 11.8, gcc 11.4
```bash
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .

cd voxel/cuda_voxelization
pip install .

cd voxel.cuda_devoxelization
pip install .
```

## Dataset

For generation, dataset `ShapeNetCore.v2.PC15k` can be downloaded [here](https://github.com/stevenygd/PointFlow), which is used in the paper [[2]DiT-3D](#references).

```
ShapeNetCore.v2.PC15k
├── Category1
│   ├── train
│   │   ├── pcd1.npy
│   │   ├── pcd2.npy
|   |    ...
|   ├── test
|   │   ├── pcd1.npy
|   │   ├── pcd2.npy
|   |    ...
│   └── val
│       ├── pcd1.npy
│       ├── pcd2.npy
|        ...
├── Category2
│   ├
 
 ...
```

For completion, dataset `ShapeNetCompletion` can be downloaded [here](https://gateway.infinitescript.com/s/ShapeNetCompletion)

```
ShapeNetCompletion
├── train
│   ├── complete
│   │   ├── category1
|   |   |    ├── pcd1.npy
|   |   |    ├── pcd2.npy
|   |   |     ...        
│   │   ├── category2
|   |        ...
│   └── partial
│       ├── category1
|       |    ├── pcd1
|       |    |    ├── 00.npy
|       |    |    |    ...
|       |    |    └── 07.npy
|       |    ├── pcd2
|       |    |    ├── 00.npy
|       |    |    |    ...
|       |    |    └── 07.npy
|       |    ├── pcd3
|       |     ...        
│       ├── category2
|        ...
├── test
│    ...
└── val
     ...
 
 ...
```


## Examples
### Training
For training the model (scale type 'DiT-S/4' - Small, patch dim 4) with single gpu, please run
```bash
python train_completion.py --distribution_type 'single' --gpu 0 \
  --dataset PATH_TO_DATASET \
  --category chair \
  --expertiment_name FILE_NAME_IN_CHECKPOINT \
  --model PATH_TO_PRETRAINED_MODEL \
  --freeze \
  --model_type 'DiT-S/4' \
  --window_size 4 --window_block_indexes '0,3,6,9' \
  --bs 16 \
  --voxel_size 32 \
  --lr 1e-4 \
  --use_tb \
  --niter FINAL_EPOCH \
  --saveIter 50 \
  --diagIter 50 \
  --vizIter 50
```

### Testing
For testing the model please run
```bash
python test_completion.py --dataroot PATH_TO_DATASET \
    --category chair --num_classes 1 \
    --experiment_name FILE_NAME_IN_CHECKPOINT \
    --niter FINAL_EPOCH \
    --bs 16 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model PATH_TO_MODEL_WEIGHTS \
    --gpu 0
```

## Trained model

Pretrained model in this project can be downloaded [here](https://drive.google.com/file/d/1rO-5-djSQPIqraG-7XSHlCBTBPN1lLbf/view?usp=drive_link).

The pretrained model from the original paper [DiT-3D](https://arxiv.org/pdf/2307.01831) can be downloaded from their [github page](https://github.com/DiT-3D/DiT-3D)

## Theory Part

### summary of the paper - [DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation](https://arxiv.org/pdf/2307.01831)

The Paper introduces a new method for 3D shape generation called DiT-3D (Diffusion Transformer for 3D shape generation). This method combines the strengths of diffusion models and Transformer architectures to generate high-quality 3D shapes (ShapeNet).Their method extends the diffusion model to the 3D domain for point cloud generation and introduces the 3D Window Attention module. It also fine-tunes using pre-trained data from Vision Transformers, significantly improving training efficiency and achieving state-of-the-art (SOTA) results in point cloud generation.

### New Contributions in This Project

The original paper implements the task of point cloud generation. (What is point cloud generation: the model receives a classification for an object, such as "car" or "chair," and the model should output a shape that matches human perception of that classification). In generative tasks, the diffusion transformer uses two types of condition embeddings at each layer: the given time step 𝑡 and the object's classification. In this project, the task is to reconstruct incomplete point clouds (i.e., to fill in missing points in an object or enhance the density of sparse point clouds). The condition for this task will shift from the object's classification to the input incomplete point cloud. 

Main contributions in this project:
  - Attempted a new embedding for incomplete point clouds based on Vision Transformer blocks.
  - Using Adaptformer fine-tune the model using the pre-existing weights from the point cloud generation tasks (as provided in the original paper).


### Diffusion model
DDPM (Denoising Diffusion Probabilistic Models) is a generative model that restores images through a step-by-step denoising process. 

<figure align="center">
    <img src="assets/ddpm.png" alt="DDPM" style="width:50%; height:auto;">
    <figcaption>Figure 2: The diagram illustrates how the Denoising Diffusion Probabilistic Model (DDPM) progressively removes noise using the 3D DiT noise predictor. Ultimately, the model generates 3D point clouds that approximate the original data from noise.</figcaption>
</figure>

Assuming there are 1000 timesteps, starting from the original point cloud, random Gaussian noise is added at each step, which called the forward process. During training, the input is the point cloud at time t, along with t and the partial point cloud as conditions. The denoising model predicts the noise, and by subtracting the predicted noise from the noisy input point cloud, and the point cloud at time t-1 is obtained, this process is called the reverse process. This is then compared to the ground truth point cloud at time t-1 obtained in the forward process to calculate the loss. In the inference phase, a fully Gaussian noise point cloud is input, and after 1000 timesteps of the reverse process, the final generated point cloud is obtained.

### Details in Denoise model

The Diffusion Transformer builds upon the Vision Transformer by incorporating adaLN-Zero. AdaLN is an extension of standard layer normalization that allows the layer normalization parameters to be dynamically adjusted based on the input data or additional conditional information (such as conditions added to the model, which can include classifications, text, or in this project, partial point clouds).

And due to the increased token length resulting from the additional dimension in 3D space, the computational cost of 3D Transformers can be significantly high. To address this issue, the paper [[2]DiT-3D](#references) introduces an efficient 3D window attention mechanism into the Transformer blocks, enabling the propagation of point-voxel features with efficient memory usage.

In this project, AdaptFormer is integrated to fine-tune the existing model weights, adapting the original generative task to a completion task. AdaptFormer is a plug-and-play, lightweight module that adds less than 2% extra parameters to the ViT, while enhancing its transferability without updating its original pre-trained parameters.

<div align="center">
    <img src="assets/model.png" alt="Image 1" style="width:70%; height:auto;">
    <!-- <img src="assets/DiT-3D.png" alt="Image 1"  style="width:25.5%; height:auto;"> -->
    <figcaption>Figure 3: The left side shows the model architecture from the original paper, while the right side presents the model architecture used in this project.</figcaption>
</div>

In terms of model architecture, aside from changes in the condition embedding, the rest remains the same as in the original paper. Refer to the embedding section in the bottom right corner of the model architecture diagram, where the original label embedding has been replaced with partial point cloud embedding, the sturcture of partial point cloud embedding is shown in the figure below.

<figure align="center">
    <img src="assets/contidion encoder.png" alt="This is an image" style="width:35%; height:auto;">
    <figcaption>Figure 4:  the sturcture of partial point cloud embedding</figcaption>
</figure>

The structure and complexity(like the number of ViT Blocks, in the project is setted as 3) of the encoder also significantly impact the final completion results. Other possible structures include encoders like those in PointNet [[3]PointNet](#references) or U-Net-based encoders. And the choice of max pooling is motivated by the fact that, as noted in PointNet, due to the unordered nature of point clouds, a symmetric function is needed to aggregate information from all points. It has been demonstrated that when the feature dimensions are sufficiently large, max pooling can approximate any symmetric function f described in the paper. However, in this project, with the use of Transformers, the point clouds are first patchified and positional information is assigned to each patch, it disrupts the unordered nature of the point clouds.. Therefore, other pooling methods could also be considered for the final step.


Finally, The model structure for fine-tuning using the Adaptformer is as follows:

<div align="center">
    <img src="assets/DiT-3D-Adaptformer.png" alt="Image 2" style="width:35%; height:auto;">
    <figcaption>Figure 5:  The model structure for fine-tuning using the Adaptformer. The left part contains an encoder-decoder structure block is Adaptformer, the rest corresponds to the model structure described on the right side of Figure 2. The layers shown in yellow contain trainable parameters, while those in blue have frozen parameters. </figcaption>
</div>

**Hyperparameter:**

*The parameters marked out are different from or newly added compared to those in the original paper. Most of the training parameters, such as optimizer parameters and loss functions, use those from the original paper*.

| Hyperparameter | Value |
|:-------------|:---------------:|
| architecture design     | DiT + 3D windown attention + `Adaptformer` + `Partial Point Cloud Condition Encoder` (12 DiT blocks in DiT, 3 ViT blocks in condition encoder)        |
| training iteration count | **1200** |
| learning rate | 1e-4 |
| diffusion steps | 1000 |
| noise scheduler | Linear |
| sampler scheduler | Uniform |
| optimizier | AdamW |
| betas for optimizer | (0.9, 0.999) |
| weight decay | 0.0 |
| dropout | 0.0 |
| loss function | MSE |
| `the point number of partial point cloud` | `512` |
| `the point number of complete point cloud` | `2048` |
| batch size | 16 |

**Sturcture of Partial Point Cloud Condition Encoder**
| Hyperparameter | Value |
|:-------------|:---------------:|
| voxelization resolution | 32 |
| numbers of BiT Blocks in Partial Point Cloud Condition Encoder | 3 |
| Pooling | Max Pooling |
| output token length | 384 |

**Sturcture of Adpatformer**
| Hyperparameter | Value |
|:-------------|:---------------:|
| input dimension of encoder| 384 |
| hidden dimension | 64 |
| output dimension of decoder | 384 |
| activate function | ReLU |

In this project, we initially trained a point cloud completion model based on the diffusion transformer. Since the number of points in the incomplete point cloud is one-fourth of the number of points in the complete point cloud, the number of ViT blocks in the condition embedding can be fewer than the number of DiT blocks in the diffusion model; here, we chose 3 ViT blocks. The pooling method is based on the max pooling used in the encoder of the PointNet model.


### Future Work / Potential Performance Improvements

1. Increase the number of ViT blocks or hidden dimension in the parital point cloud condition encoder, but having too many blocks does not necessarily lead to better performance.
2. Use different pooling methods in the parital point cloud condition encoder.
3. Try different encoder architectures.
4. Explore fine-tuning methods other than AdaptFormer, as AdaptFormer may not necessarily perform the best in diffusion models. Refer to the following articles: [[4]DiffFit](https://arxiv.org/pdf/2304.06648)

## Result

### Traninig

During training, the model weights were saved every 50 epochs. The figure below shows the performance of the model on the validation dataset at each saved checkpoint, with the metric being the 1-NNA Chamfer Distance.

<figure>
    <img src="assets/train.png" alt="This is an image">
    <figcaption>Figure 6: Recorded 1-NNA-CD values during training process. The final model weights are selected based on the epoch with the lowest 1-NNA-CD value. Blue line: train with using Adaptformer Fine-Tune. Green line: train with Global Fine-Tune (all the parameter is trainable) </figcaption>
</figure>

We can observe that during global fine-tuning, the training reached the 300th epoch, the Chamfer Distance did not decrease, the model's performance did not improve.

In the model using adaptformer, we can see that the performance of the model was very noticeable imporoved. The Chamfer Distance decreased significantly from close to 1 at the beginning to a minimum of 0.20. Ultimately, the model weights with the lowest Chamfer Distance were selected for subsequent testing.

The training has currently reached 1200 iterations, starting fine-tuning from the weights at epoch 7650 of the original paper to epoch 8850. For the new added dataset for completion tasl `ShapeNetCompletion` , each complete point cloud corresponds to 8 partial point clouds. If the dataloader outputs the partial and complete point clouds in a one-to-one manner, the memory requirements for the dataloader are very high. In this project, a distributed training approach is used, where each data loader loads only one partial point cloud at a time and trains for 300 epochs (for example, the data loader contains only the partial point cloud with index 00 and its corresponding complete point cloud. The dataset structure is detailed [here](#dataset).). In this project, training was conducted for 300 epochs each for indices 00, 02, 04, and 06 of the partial point cloud.

However, this training approach reduces the computational load on the computer but also result in decreased model accuracy. Additionally, training for 1200 epochs does not ensure that the model has fully converged, and further fine-tuning may still improve the model's accuracy.

<!-- - [ ]TODO finetune using DiffFit ([paper]("https://arxiv.org/pdf/2304.06648)) -->

**Evaluation on test dataset - epoch 8649:**

*The evaluation data is in the format of mean/variance, mean and variance calculated under four different seeds - 0, 1, 42, 123*

| CD Average⭣ | EMD Average⭣ | f1 Average⭣| 1-NNA-CD⭣ | 1-NNA-EMD⭣ | COV-CD⭡ | COV-EMD⭡ |
|:-------------|:---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 0.0032/2.250e-08 |55.97/1.194e+00 | 0.6294/4.875e-05 | 0.4386/3.7550e-04         | 0.4928/4.268e-03      |  0.5842/1.135e-04          | 0.5772/1.163e-03          | 

<!-- **Evaluation on validation dataset - epoch 8649:**
|CD Average⭣ |EMD Average⭣ | f1 Average⭣ | 1-NNA-CD⭣ | 1-NNA-EMD⭣ | COV-ECD⭡ | COV-EMD⭡ |
|:-------------|:---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 0.0027 | 60.0288 | 0.6939 | 0.20     | 0.38         | 0.84        | 0.60        | -->

**Evaluation on test dataset with sparse point cloud - epoch 8649:**

*The evaluation data is in the format of mean/variance, mean and variance calculated under four different seeds - 0, 1, 42, 123*

|CD Averagee⭣ | EMD Average⭣ | f1 Average⭣ | 1-NNA-CD⭣ | 1-NNA-EMD⭣ | COV-CD⭡ | COV-EMD⭡ |
|:-------------|:---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 0.0027/6.667e-09 | 53.10/1.793e+00 |0.6802/7.775e-05| 0.4023/7.254e-04         | 0.4641/3.805e-04       | 0.6274/1.532e-03          | 0.5859/3.954e-04          |

(⭣ means the lower the better, ⭡ means the higher the better)
*(Sparse point clouds are generated by randomly sampling fewer points from the complete point clouds in the test set. In the `ShapeNetCompletion` dataset, partial point clouds are the result of removing a part of the object, but sparse point clouds consist of the entire shape of the object but with very few points.)*


From the data, it is evident that this result has a significant performance gap compared to the state-of-the-art (SOTA).

<!-- - [ ]TODO Find a suitable baseline to compare the performance gap. -->

In Paper [SpareNet](https://arxiv.org/pdf/2103.02535), their average CD value for the chair category is 0.000616.

### Detailed Case Analysis and visualization

#### Cases from test set:

|  |  |  |
|---------|---------|---------|
| ![bathub condition](assets/example1.png) | ![bathub predicted](assets/example2.png) | ![bathub ground truth](assets/example3.png) |
| ![couch condition](assets/example4.png) | ![couch prediction](assets/example5.png) | ![couch ground truth](assets/example6.png) |
| ![bed condition](assets/example7.png) | ![bed prediction](assets/example8.png) | ![bed ground truth](assets/example9.png) |
| ![bathub condition](assets/example10.png) | ![bathub predicted](assets/example11.png) | ![bathub ground truth](assets/example12.png) |
| ![couch condition](assets/example13.png) | ![couch prediction](assets/example14.png) | |

The examples above are from the dataset, with the left side showing the point clouds reconstructed by the model and the right side displaying the ground truth. It can be observed that the model possesses a basic ability to restore partial point clouds. The model can distinguish the shapes of objects such as the structure of a chair, including its four legs and backrest (e.g., the first row, first column), the driver's seat of a car (e.g., the third row, third column), a sofa (e.g., the fourth row, second column), and a swivel chair (e.g., the fifth row, first column). And the reconstructed shapes are distinguishable and align with logical expectations.

For basic chair structures (such as those in the first, second, and the first two columns of the third rows), the model's reconstruction is more accurate than for other shapes. However, there are still significant deviations in the details. For example, the chair in the second row, second column, as well as the chairs in the third row, second column, and the fifth row, second column, show discrepancies in the legs compared to the ground truth and fail to cover the input partial point cloud. The car driver's seat only reconstructs a rough shape, missing most of the finer details. The chair in the fourth row, third column, contains structural errors in its reconstruction, and the swivel chair in the fifth row, first column, has incorrect structural reconstruction in the base.

### Model Performance under Different Input Point Counts

<figure>
    <img src="assets/input points ablation.png" alt="This is an image">
    <figcaption>Figure 7: The figure shows the performance of different numbers of points from an input partial point cloud, as measured by Chamfer Distance. The x-axis represents the number of points, ranging from 256 to 1792, while the y-axis shows the Chamfer Distance.</figcaption>
</figure>

The number of points in the partial point cloud during training is 512. When the number of points in the input partial point cloud is 256 or 768, the reconstruction quality remains still optimal. However, when the number of points exceeds 1024, the reconstruction quality significantly deteriorates.

This may seem counterintuitive, as the reconstruction accuracy decreases when more points are input (i.e., when more information is provided). This is because the partial point cloud is only used as a conditional guidance input to the condition embedding, and the final output point cloud is reconstructed from pure noise via the diffusion model, rather than being directly based on the partial point cloud. As an optimization, noise can be added to the input partial point cloud to reach the required number of points, and this can be used as the input noise for the diffusion model, thereby incorporating the original input data. Since this project fine-tunes a model based on pre-existing task-specific weights, the diffusion model still operates by transitioning from pure noise to the final output to maintain consistency.

However, our final goal is still to reconstruct the complete object using the fewest number of points from the partial point cloud. Based on the results from Figure 7, when the number of input points is very small, the reconstruction quality does not degrade.

<!-- #### different levels sampling on partial point cloud
- [ ]TODO

#### different levels sampling on sparse point cloud
Sampling sparse point clouds with different levels of density from the ground truth was used as input conditions for the model to observe the impact of different point numbers on the final point cloud reconstruction quality, using Chamfer Distance Average.

**case 1** 

![different levels sampling 1](assets/plot_case1.png) -->



<!-- #### The diversity of reconstructions generated by the model
- [ ]TODO -->



<!-- ## Project status
June:
- [x] Startup: try SemanticKITTI Dataset, and diffuison model training
  - [x] to document(**Done**: Understand and try some programming on the datasets ShapeNet and KITTI360-test-demo. SemanticKITTI (80GB) is too large to download on my own computer. Use Open3D for visualization and PyntCloud for voxelization.) `13.06`
  - [ ] ? voxelization module from DiT3D ( actually from PVCnn)

- [ ] TODO - repoduce the results from the two stduies\
`Device issues can't be resolved, linux system required (PytorchEMD, MinkowskiEngine, pytorch3d)`
  - -[x] TODO [code[1]](https://github.com/PRBonn/LiDiff)
    - the [result](tryout/Lidiff/) from example pipeline 
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
The code is tested on Ubuntu 24.04, python 3.10, pytroch 2.1.2, CUDA 11.8. -->


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

3. [Qi C R, Su H, Mo K, et al. Pointnet: Deep learning on point sets for 3d classification and segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 652-660.](https://arxiv.org/pdf/1612.00593)

4. [Xie E, Yao L, Shi H, et al. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 4230-4239.](https://arxiv.org/pdf/2304.06648)