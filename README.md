# VAE-classifier
**End to End Image classification and Compression with Variational Auto Encoders**, in Preprint of IEEE Internet of Things Journal.
This repo contains codes for paper.

## Joint image classification and compression with VAEs
To overcome the infrastructural barrier of limited network 
bandwidth in cloud ML, existing solutions 
have mainly relied on traditional compression codecs such
as JPEG that were historically engineered for human-end users instead of ML algorithms. 
Traditional codecs do not necessarily preserve 
features important to ML algorithms
under limited bandwidth, leading to 
potentially inferior performance. This work investigates
application-driven optimization of programmable
commercial codec settings for
networked learning tasks such as image classification. 
Based on the foundation of variational autoencoders (VAEs), we 
develop an end-to-end networked 
learning framework by jointly optimizing 
the codec and classifier without 
reconstructing images for given data rate (bandwidth). Compared with standard JPEG codec, the proposed VAE joint compression and classification framework achieves classification accuracy improvement by over 10\% and 4\%, respectively, for CIFAR-10 and ImageNet-1k data sets
at data rate of 0.8 bpp. Our proposed VAE-based models show 65%-99% reductions in encoder size, x1.5-x13.1 improvements in inference speed and 25%-99% savings in power compared to baseline models. We further show that a simple decoder can reconstruct imageswith 
sufficient quality without compromising classification accuracy.

<p align="left">
  <img src="https://github.com/chamain/VAE-classifier/blob/master/imgs/fullModel.png" width="700" title="plane">
</p>

Overview of the proposed VAE classifier during inference:
Quantized latent vector $\mathbf{\hat z}$ is encoded into bit stream by a context-adaptive arithmetic encoder (AE) assisted by probability estimator (PE). At receiver, probability of each symbol (shown in cyan) is estimated by using a learned PE based on previously decoded latents (shown in gray). Without groundtruth distribution of the latent elements $q_{\boldsymbol{\phi}}(\mathbf{\hat z}|\boldsymbol{x})$ at the receiver, PE learns to approximate $p_{\boldsymbol{\theta}} \approx q_{\boldsymbol{\phi}}$ during training.

## Results
<p align="left">
  <img src="https://github.com/chamain/VAE-classifier/blob/master/imgs/vae_results.png" width="700" title="plane">
</p>

Classification accuracy vs rate results for end-to-end compression and classification on (a) CIFAR-10 and (b) CIFAR-100 data sets. The proposed VAE
based compression and classification framework outperforms popular commercial image compression codecs in terms of rate-accuracy, at lower bandwidths.

<p align="left">
  <img src="https://github.com/chamain/VAE-classifier/blob/master/imgs/imagenet_low.png" width="500" title="plane">
</p>

Classification accuracy vs rate on ImageNet-1k for end-to-end
compression and classification. The proposed VAE based compression and
classification framework (AE-V4) significantly outperforms JPEG commercial
image compression codecs in terms of rate-accuracy.
## Citation
If you find our work useful in your research, please consider citing:
```
@article{chamain2022end,
  title={End-to-End Image Classification and Compression with variational autoencoders},
  author={Chamain, Lahiru D and Qi, Siyu and Ding, Zhi},
  journal={IEEE Internet of Things Journal},
  year={2022},
  publisher={IEEE}
}
```
