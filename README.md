# Uformer for Diffraction Denoising

## Original repo is: [[Uformer: A General U-Shaped Transformer for Image Restoration (CVPR 2022)]](https://github.com/ZhendongWang6/Uformer)

## Environmental setup
```bash
conda create -n DiffractionDenoising python==3.10
conda activate DiffractionDenoising
git clone https://github.com/Graph4HEP/Diffraction_denoising.git
cd Diffraction_denoising
pip install -r requirements.txt
```

## Data preparation 
The example tar.gz data is located at [here](example_data/)

Unzip the tar.gz files by:
```bash
cd example_data
tar -xvf training.tar.gz
tar -xvf validation.tar.gz
```

The training and validatation data both contains 2 folders (LC and HC). 

The LC folder contains the input noising data.

The HC folder contains the target denoising data.

## Training
To train Uformer for denoising, you can begin the training by:

```sh
sh script/train_denoise.sh
```

The script already define the parameters used in the model.

## Test
To test the model, run the following commands:
```bash
cd test
python test.py log_dirs data_dir_which_contains_LC_and_HC
```

## Computational Cost

The original repo provide a simple script to calculate the flops by ourselves, a simple script has been added in `model.py`. You can change the configuration and run:

```python
python3 model.py
```

## Citation
If you find this project useful in your research, please consider citing the original paper:

```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhendong and Cun, Xiaodong and Bao, Jianmin and Zhou, Wengang and Liu, Jianzhuang and Li, Houqiang},
    title     = {Uformer: A General U-Shaped Transformer for Image Restoration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17683-17693}
}
```

and our study:
```
@misc{diffraction_denoising_github,
  author = {Bingzhi, Li},
  title = {Diffraction Denoising},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {Accessed: \url{https://github.com/Graph4HEP/Diffraction_denoising}},
}
```
