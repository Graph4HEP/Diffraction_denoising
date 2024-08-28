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

Then generate training patches for training by:
```bash
cd example_data
tar -xvf training.tar.gz
tar -xvf validation.tar.gz
```

## Training
### Denoising
To train Uformer, you can begin the training by:

```sh
sh script/train_denoise.sh
```

## Evaluation

## Computational Cost

We provide a simple script to calculate the flops by ourselves, a simple script has been added in `model.py`. You can change the configuration and run:

```python
python3 model.py
```

> The manual calculation of GMacs in this repo differs slightly from the main paper, but they do not influence the conclusion. We will correct the paper later.


## Citation
If you find this project useful in your research, please consider citing the original paper and our study:

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

