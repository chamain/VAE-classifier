# AE-classifier: End to End Image classification and Compression with Variational Auto Encoders
python project files for the [paper](https://www.researchgate.net/profile/Lahiru_Dulanjana_Chamain_Hewa_Gamage/publication/334997996_Quannet_Joint_Image_Compression_and_Classification_Over_Channels_with_Limited_Bandwidth/links/5d54911792851c93b630b715/Quannet-Joint-Image-Compression-and-Classification-Over-Channels-with-Limited-Bandwidth.pdf) - Quannet: Joint Image Compression and Classification Over Channels with Limited Bandwidth

1. This project uses CIFAR-10 dataset and we provide the python code for the end to end training of QuanNet in **keras**.
2. We use **ResNet-20** as the classification network
3. We use the following simple trainable layer for QuanNet.
![alt text][logo]

[logo]: https://github.com/chamain/QuanNet/blob/master/images/quanblock.PNG "Quan block"

## Installation
Download the python scripts from the repository.
Install the following dependencies.
```
pip install PyWavelets
```
## Training
```
python train.py
```
## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{chamain2019quannet,
  title={Quannet: Joint Image Compression and Classification Over Channels with Limited Bandwidth},
  author={Chamain, Lahiru D and Cheung, Sen-ching Samson and Ding, Zhi},
  booktitle={2019 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={338--343},
  year={2019},
  organization={IEEE}
}
```
Patent pending.
