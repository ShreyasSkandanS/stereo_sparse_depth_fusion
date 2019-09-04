# Stereo and Sparse Depth Fusion

This is the corresponding code repository for our paper *Real Time Dense Depth
Estimation by Fusing Stereo with Sparse Depth Measurements*

## Reference

If you use this work, please cite our
[paper](https://ieeexplore.ieee.org/abstract/document/8794023):

```
@inproceedings{shivakumar2019real,
  title={Real time dense depth estimation by fusing stereo with sparse depth measurements},
  author={Shivakumar, Shreyas S and Mohta, Kartik and Pfrommer, Bernd and Kumar, Vijay and Taylor, Camillo J},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={6482--6488},
  year={2019},
  organization={IEEE}
}
```

In this repository, we also use the Semi Global Matching algorithm implementation by Koichiro Yamaguchi, therefore if you use this repository, please also cite the [original paper](http://ttic.uchicago.edu/~dmcallester/SPS/index.html) by Koichiro et al. and follow their original license agreements.
```
@inproceedings{yamaguchi2014efficient,
  title={Efficient joint segmentation, occlusion labeling, stereo and flow estimation},
  author={Yamaguchi, Koichiro and McAllester, David and Urtasun, Raquel},
  booktitle={European Conference on Computer Vision},
  pages={756--771},
  year={2014},
  organization={Springer}
}
```

## Dependencies

1. OpenCV (Tested on 3.X.X)
2. libpng++ (Tested on 0.2.5-1)
```
sudo apt-get install libpng++-dev
```
3. OpenMP (optional)
4. CMake


## Testing Code

Navigate to the main code repository
```
mkdir build
cd build
cmake ..
make -j
```

You can run our KITTI example as follows
```
./stereo_depth_fusion
```

You should now see the following images in the *results* folder:
* *sgm_default.png* - Output of SGM algorithm
![sgm](/results/sgm_default.png)
* *sparse_mask.png* - Mask of ground truth disparities used
![sparsemask](/results/sparse_mask.png)
* *fuse_naive.png* - Output of our naive fusion implementation
![naive](/results/fuse_naive.png)
* *fuse_neighborhoodsupport.png* - Neighborhood Support Method
![neighborhood](/results/fuse_neighborhoodsupport.png)
* *fuse_diffusionbased.png* - Diffusion Based Method
![diffusion](/results/fuse_diffusionbased.png)


## Understanding The Code

The entire Semi Global Matching implementation is from Koichiro Yamaguchi and
lives in the *SGMStereo.cpp* and *SGMStereo.h* files. The adjustments to the
cost volumes are made in *[SGMStereo.cpp](SGMStereo.cpp)* in the following functions:
* *updateCostVolume_NRF()* - Naive Fusion
* *updateCostVolume_NS()*  - Neighborhood Support
* *updateCostVolume_DB()*  - Diffusion Based Method

These methods are called by a helper script
*[stereo_depth_fusion.cpp](stereo_depth_fusion.cpp)* and is tested on a pair of
randomly chosen [KITTI
2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) images.

## Results

#### KITTI 2015
*From top to bottom:*
* Input color image (left)
* Semi Global Matching (original)
* Neighborhood Support Method *(ours)*
* Diffused Based *(ours)*
* Anisotropic Diffusion
* Ground Truth (LiDAR)

*Images:*

![](/imgs/kitti_1.png) ![](/imgs/kitti_2.png)

#### Middlebury
*From left to right:*
* Input color image (left)
* Semi Global Matching (original)
* Neighborhood Support Method *(ours)*
* Diffusion Based *(ours)*
* Anisotropic Diffusion
* Ground Truth

*Images:*

![](/imgs/mb_1.png) ![](/imgs/mb_2.png)

#### PMD Monstar Dataset *(internal)*
*From left to right:*
* Input color image (left)
* Semi Global Matching (original)
* Neighborhood Support Method *(ours)*
* Diffusion Based *(ours)*
* Anisotropic Diffusion
* Mask of Time-of-Flight depth measurements

*Images:*

![](/imgs/mon_1.png) ![](/imgs/mon_2.png)

If you have any questions regarding this repository or our method feel free to
raise an issue request in this repository or email me directly.

Note: this repository is part of a larger on-going research project, and
therefore contains multiple variables that are unused (but are used in following
research). I've tried to add ignores to most of them, but please excuse these if
they occur. Feel free to make pull requests if you notice any bugs or wish to
cleanup this code.


