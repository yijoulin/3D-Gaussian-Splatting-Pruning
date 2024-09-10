## Overview
In this project, I developped two new function to original 3DGS model. 
1. Gaussian Redundancy 
Remove redundant gaussian by the first method in [Reducing the Memory Footprint of 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/reduced_3dgs/), but without using threshold. 
The removed result of using LEGO dataset of [NeRF Synthetic dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) is shown as follow table.
|    | Number of Gaussians   | SSIM   | PSNR|LPIPS|
| ------- | ------- | ------- |
| Original Baseline   | 346636   | 0.982594   |36.0502|0.015989|
| With Pruning  |233482| 0.982491   |36.00119|0.016171|

2. Showing the deleted 3D Gaussians on the reconstructed scene by labelled 3D Gaussian. The result is displayed by removed gaussian in human head Avatar. ![Result](assets/result.png)


## Environment
* gcc == 7.1.0
* cuda == 1.6.2
* colmap
* Viewers

## Intruction
Follow all the setup steps in [original 3DGS tutorial](https://github.com/graphdeco-inria/gaussian-splatting.git)
