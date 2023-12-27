# dataset

1. IAT LOL-V1
   train 485, test 15
2. BrighteningTrain dataset
   train 1000
3. SICE dataset
   train 300
4. our use
   (IAT LOL-V1) + (BrighteningTrain dataset) + (SCIE dataset)
   (400 : 100) + (800 : 200) + (240 + 60) == (1440 : 360)

# 1500

(IAT LOL-V1) + (BrighteningTrain dataset)
(400 : 100) + (800 : 200) == (1200 : 300)

# 1800

(IAT LOL-V1) + (BrighteningTrain dataset) + (SCIE dataset)
(400 : 100) + (800 : 200) + (240 + 60) == (1440 : 360)

# IAT in Low-Level Vision

## I. Low-Light Enhancement (LOL-V1 dataset, 485 training image, 15 testing image)

1. Download the dataset from the [here](https://daooshee.github.io/BMVC2018website/). The dataset should contains 485 training image and 15 testing image, and should format like:

```
Your_Path
  -- our485
      -- high
      -- low
  -- eval15
      -- high
      -- low
```

2. Evaluation pretrain model on LOL-V1 dataset

```
python evaluation_lol_v1.py --img_val_path Your_Path/eval15/low/
```

Results:
| | SSIM | PSNR | enhancement images |
| -- | -- | -- | -- |
| results | **0.809** | **23.38** | [Baidu Cloud](https://pan.baidu.com/s/1M3H5coIOwfzYdTbZCkM42g) (passwd: 5pj2), [Google Drive](https://drive.google.com/drive/folders/1fgDUEbdiRkLbORZt4LMTX5rFB_erexOc?usp=sharing)|

3. Training your model on LOL-V1 dataset (get our closely result).

# patch version

Step 1: crop the LOL-V1 dataset to 256 $\times$ 256 patches:

```
python LOL_patch.py --src_dir Your_Path/our485 --tar_dir Your_Path/our485_patch
```

Step 2: train on LOL-V1 patch images:

```
python train_lol_v1_patch.py --img_path Your_Path/our485_patch/low/ --img_val_path Your_Path/eval15/low/
```

Step 3: tuned the pre-train model (in Step 2) on LOL-V1 patches on the full resolution LOL-V1 image:

```
python train_lol_v1_whole.py --img_path Your_Path/our485/low/ --img_val_path Your_Path/eval15/low/ --pretrain_dir workdirs/snapshots_folder_lol_v1_patch/best_Epoch.pth
```

# original image version

Step 1: train on LOL-V1 original images

```
python train_lol_v1_whole.py --img_path Your_Path/our485/low/ --img_val_path Your_Path/eval15/low/
```

<br/>

## II. Low-Light Enhancement (LOL-V2-real dataset, 589 training image, 100 testing image)

1. Download the dataset from [Baidu_Cloud](https://pan.baidu.com/s/1Md5r4Lup8NVQI2ixKTIlGQ)(passwd: m7f7) or [Google Drive](https://drive.google.com/file/d/17UiWwwLHHveHf7N2Ubknpk7FUsN06W6a/view?usp=sharing), the dataset should format like:

```
Your_Path
  -- Train
      -- Normal
      -- Low
  -- Test
      -- Normal
      -- Low
```

2. Evaluation pretrain model on LOL-V2-real dataset

```
python evaluation_lol_v2.py --img_val_path Your_Path/Test/Low/
```

Results:

|         | SSIM      | PSNR      | enhancement images                                                                                                                                                                 |
| ------- | --------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| results | **0.824** | **23.50** | [Baidu Cloud](https://pan.baidu.com/s/1XH8Bpo0UgrJEqz_gOefiQA)(passwd: 6u3m), [Google Drive](https://drive.google.com/drive/folders/1rxBGGLIguNP0r_Of4dxQ1VAZRnGYJZGu?usp=sharing) |

3. Training your model on LOL-V2-real dataset (single GPU), for LOL-V2-real, you don't need create patch and directly train is OK.

```
python train_lol_v2.py --gpu_id 0 --img_path Your_Path/Train/Low --img_val_path Your_Path/Test/Low/
```

<br/>

## Others:

1. To use the model for a image enhancement demo show, direct run:

```
python img_demo.py --file_name demo_imgs/low_demo.jpg --task enhance
```

2. To check how many parameters in IAT model, direct run:

```
python model/IAT_main.py
```

Dataset Citation:

```
@inproceedings{LOL_dataset,
  title={Deep Retinex Decomposition for Low-Light Enhancement},
  author={Chen Wei and Wenjing Wang and Wenhan Yang and Jiaying Liu},
  booktitle={British Machine Vision Conference},
  year={2018},
}

@InProceedings{Exposure_2021_CVPR,
    author    = {Afifi, Mahmoud and Derpanis, Konstantinos G. and Ommer, Bjorn and Brown, Michael S.},
    title     = {Learning Multi-Scale Photo Exposure Correction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition },
    year      = {2021},
}
```
