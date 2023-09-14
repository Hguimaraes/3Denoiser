# 3Denoiser

[Link](https://www.isca-speech.org/archive/l3das_2022/guimaraes22_l3das.html) to the paper

## Cite

```latex
@inproceedings{guimaraes22_l3das,
  author = {Guimarães, Heitor R. and Beccaro, Wesley and Ramirez, Miguel A.},
  title = {{A Perceptual Loss Based Complex Neural Beamforming for Ambix 3D Speech Enhancement}},
  year = {2022},
  booktitle = {Proc. L3DAS22: Machine Learning for 3D Audio Signal Processing},
  pages = {16--20},
  doi = {10.21437/L3DAS.2022-4},
}
```
## Abstract

> This work proposes a novel approach to B-Format AmbiX 3D speech enhancement based on the short-time Fourier transform (STFT) representation. The model is a Fully Complex Convolutional Network (FC2N) that estimates a mask to be applied to the input features. Then, a final layer is responsible for converting the B-format to a monaural representation in which we apply the inverse STFT (ISTFT) operation. For the optimization process, we use a compounded loss function, applied in the time-domain, based on the short-time objective intelligibility (STOI) metric combined with a perceptual loss on top of the wav2vec 2.0 model. The approach is applied on Task 1 of the L3DAS22 challenge, where our model achieves a score of 0.845 in the metric proposed by the challenge, using a subset of the development set as reference.

## Model architecture

@TODO

## How to run

@TODO
