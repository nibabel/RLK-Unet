# RLK_Unet
The RLK_Unet deep learning model that segments brain metastasis (BMs).

<img width="1290" alt="RLK_Unet model image" src="https://github.com/nibabel/rlk_unet/assets/135964734/5dfd8827-ed35-4204-9996-c20256940071">

<img width="1715" alt="Result" src="https://github.com/nibabel/rlk_unet/assets/135964734/bf9ac8c0-5671-4d8d-aad7-0f7d87fe628d">




If you use RLK-Unet, please cite the following paper:

"Son S, Joo B, Park M, Suh SH, Oh HS, Kim JW, Lee S, Ahn SJ, Lee J-M. Development of RLK-Unet: A clinically favorable deep learning algorithm for brain metastasis detection and treatment response assessment. Frontiers in Oncology, 13, 1273013."

https://doi.org/10.3389/fonc.2023.1273013



---

1. .
```
    git clone https://github.com/nibabel/RLK_Unet.git
```
2. .
```
    cd RLK_Unet
```
3. .
```
    pip install -e .
```
4. .
```
   rlk_unet -i {input} -gpu {GPU_device_num}
```
---
