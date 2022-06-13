# VAE-classifier
**End to End Image classification and Compression with Variational Auto Encoders**, in Preprint of IEEE Internet of Things Journal.
This repo contains codes for paper.

## Joint image classification and compression with VAEs
To overcome the infrastructural barrier of limited network bandwidth in cloud ML, existing solutions have mainly relied on traditional compression codecs such as JPEG that were historically engineered for human- end users instead of ML algorithms. Traditional codecs do not necessarily preserve features important to ML algorithms under limited bandwidth, leading to potentially inferior performance. This work investigates application-driven optimization of pro- grammable commercial codec settings for networked learning tasks such as image classification. Based on the foundation of variational autoencoders (VAEs), we develop an end-to-end networked learning framework by jointly optimizing the codec and classifier without reconstructing images for given data rate (bandwidth). Compared with standard JPEG codec, the proposed VAE joint compression and classification framework achieves classification accuracy improvement by over 10% and 4%, respectively, for CIFAR-10 and ImageNet-1K data sets at data rate of 0.8 bpp. We further show that a simple decoder can reconstruct images with sufficient quality without compromising classification accuracy.

<p align="left">
  <img src="https://github.com/chamain/VAE-classifier/blob/master/imgs/fullModel.png" width="500" title="plane">
</p>

Overview of the proposed VAE classifier during inference:
Quantized latent vector $\mathbf{\hat z}$ is encoded into bit stream by a context-adaptive arithmetic encoder (AE) assisted by probability estimator (PE). At receiver, probability of each symbol $\hat{z}_{i}$ (shown in cyan) is estimated by using a learned PE based on previously decoded latents $\hat{z}_{i-1},\cdots,\hat{z}_{1}$ (shown in gray). Without groundtruth distribution of the latent elements $q_{\boldsymbol{\phi}}(\mathbf{\hat z}|\boldsymbol{x})$ at the receiver, PE learns to approximate $p_{\boldsymbol{\theta}} \approx q_{\boldsymbol{\phi}}$ during training.

## Results

## Citation
If you find our work useful in your research, please consider citing:
```
```
