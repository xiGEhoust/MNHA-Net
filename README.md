# MNHA-Net

Official repository for the Information Security Conference (ISC) 2026 paper  *MNHA-Net: Mitigating Feature Dilution in Image Manipulation Localization via Hybrid Architecture and Late Noise Fusion*

<img src="images\Fig. 2.PNG" style="zoom:80%;" />

## Environment

* Python 3.10

* PyTorch 2.4

* CUDA 11.5 + cudnn 8.4.0

And install other by using:

```python
pip install -r requirements.txt
```

* Download the pretrained weights from [Google Drive](https://drive.google.com/file/d/1XPjAXhDS6nGNXb11VzgdNqEeM7X0p2t9/view) for train and test



## About Dataset

Training on CAT-Net joint dataset


 - CASIA2.0
 - FantasticReality_v1
 - IMD_20  
 - tampCOCO

Refer to [CAT-Net](https://github.com/mjkwon2021/CAT-Net) and [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) for more detailed.



## Train

```bash
sh train.sh
```



## Test

```python
python main_test.py
```



## Metrics

```python
python metric5.py
```



## Visualizations

<img src="images\Fig. 5.PNG" style="zoom:80%;" />
