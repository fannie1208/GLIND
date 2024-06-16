# GLIND

The official implementation for ICML2024 paper "Learning Divergence Fields for Shift-Robust Graph Representations"

Related material: [[Paper](https://arxiv.org/pdf/2406.04963)], [[Blog]()]

## Acknowledgement

The implementation of training pipeline for `observed` (Arxiv and Twitch) is based on [EERM](https://github.com/qitianwu/GraphOOD-EERM).

The implementation of training pipeline for `unobserved` (STL and CIFAR) is based on [DIFFormer](https://github.com/qitianwu/DIFFormer).

## What's news

[2024.06.03] We release the code for our model on real-world datasets that involve Observed Geometries, Partially Observed Geometries and Unobserved Geometries. More detailed info will be updated soon.

[2024.06.04] We upload the datasets (Arxiv, Twitch, Cifar, STL, DPPIN).

## Datasets

One can download the datasets (Arxiv, Twitch, Cifar, STL, DPPIN) from the google drive link below:

https://drive.google.com/drive/folders/1LctHB8_8fRqp3jq9kU3DryHXwA5PCihC

## Model and Results

We propose a geometry diffusion model that is optimized by a new learning objective (comprised of a supervised term and a regularization term) for the generalization problem with interdependent data.

![image](https://github.com/fannie1208/GLIND/assets/89764090/0240e933-a4b3-483e-9fff-8174677c83e9)

The following tables present the results of generalization with different data geometries.

### Generalization with Observed Geometries.

![image](https://github.com/fannie1208/GLIND/assets/89764090/ea2c785b-7011-4d04-b8c8-5a328d33f984)

### Generalization with Partially Observed Geometries.

![image](https://github.com/fannie1208/GLIND/assets/89764090/e7a18465-22e0-4c25-aaac-41bf2bb687b8)

### Generalization with Unobserved Geometries.

![image](https://github.com/fannie1208/GLIND/assets/89764090/89313ec2-54bc-47e5-8dba-61ffa62f7f17)


## Dependence

Python 3.8, PyTorch 1.13.0, PyTorch Geometric 2.1.0, NumPy 1.23.4

## Run the codes

Please refer to the bash script `run.sh` in each folder for running the training and evaluation pipeline on different datasets.

### Citation

If you find our code and model useful, please consider citing our work. Thank you!

```bibtex
      @inproceedings{wu2024glind,
      title = {Learning Divergence Fields for Shift-Robust Graph Representations},
      author = {Qitian Wu and Fan Nie and Chenxiao Yang and Junchi Yan},
      booktitle = {International Conference on Machine Learning (ICML)},
      year = {2024}
      }
```
