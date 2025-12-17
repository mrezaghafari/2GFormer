# Point-Cloud-Transformer

This repository contains a self-attention model that integrates sparse convolutions, specifically designed for point cloud coding tasks. The model is inspired by [**Point Transformer V2**](https://arxiv.org/abs/2210.05666) and [**PCGFormer**](https://ieeexplore.ieee.org/document/10008892), but has been modified and adapted to better suit the requirements of point cloud coding.

Sparse convolutions are obtained from [**MinkowskiEngine**](https://github.com/NVIDIA/MinkowskiEngine), which allows for efficient handling of large-scale point clouds. The novel self-attention mechanism captures the long-range dependencies between points using point features. This model aims to improve the performance of point cloud compression, segmentation, and other related tasks.

### Key Novel Components:
* Differential Positional Embedding
* Relational Scoring
* Sparsemax

### Input:
- A sparse tensor with point coordinates `[n, 3]` and point features `[n, d]`.
- The attention mechanism is designed to generate attended features.

Feel free to use the code, and if you find it useful, please cite our work:

**Paper**: [Point Cloud Geometry Coding with Relational Neighborhood Self-Attention](https://ieeexplore.ieee.org/abstract/document/10743834)
```M. Ghafari, A. F. R. Guarda, N. M. M. Rodrigues and F. Pereira, "Point Cloud Geometry Coding with Relational Neighborhood Self-Attention," 2024 IEEE 26th International Workshop on Multimedia Signal Processing (MMSP), West Lafayette, IN, USA, 2024, pp. 1-6, doi: 10.1109/MMSP61759.2024.10743834.```

**Paper**: [Scalable Graph-Guided Transformer for Point Cloud Geometry Coding](https://ieeexplore.ieee.org/abstract/document/11123804)
```M. Ghafari, A. F. R. Guarda, N. M. M. Rodrigues and F. Pereira, "Scalable Graph-Guided Transformer for Point Cloud Geometry Coding," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2025.3598605. ```

