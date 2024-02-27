# RLK-Unet
RLK-Unet: Deep Learning Model for Brain Metastasis (BMs) Segmentation

<img width="1290" alt="RLK_Unet model image" src="https://github.com/nibabel/rlk_unet/assets/135964734/5dfd8827-ed35-4204-9996-c20256940071">

<img width="1715" alt="Result" src="https://github.com/nibabel/rlk_unet/assets/135964734/bf9ac8c0-5671-4d8d-aad7-0f7d87fe628d">




If you use RLK-Unet, please cite the following paper:

"Son S, Joo B, Park M, Suh SH, Oh HS, Kim JW, Lee S, Ahn SJ, Lee J-M. Development of RLK-Unet: A clinically favorable deep learning algorithm for brain metastasis detection and treatment response assessment. Frontiers in Oncology, 13, 1273013."

https://doi.org/10.3389/fonc.2023.1273013



---

1. Clone the repository to use RLK-Unet.
```
    git clone https://github.com/nibabel/RLK_Unet.git
```
2. Go to the installed repository directory.
```
    cd RLK_Unet
```
3. Install the virtual environment package to use RLK-Unet.
```
    pip install -e .
```
4. Once the installation is complete, BMs can be segmented using the 'rlk_unet' command from anywhere in Linux.
```
   rlk_unet -i {input}
```
---


__CAUTION: Input is only 3D black blood (BB) T1-weighted image.__
