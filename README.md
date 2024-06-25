# Structured Gradient-based Interpretations via Norm-Regularized Adversarial Training
Implementation for CVPR 2024 paper [Structured Gradient-based Interpretations via Norm-Regularized Adversarial Training](http://arxiv.org/abs/2404.04647) by Shizhan Gong, [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html), and [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/).

## Sample Results
![Alt text](asset/samples.png?raw=true "Title")

## Setup
Our code is easily runable on most pytorch environment without special dependency. Or readers can install the environment via
`
pip install -r requirements.txt
`.

## Dataset
### Imagenette
The data can be downloaded from the [official page](https://github.com/fastai/imagenette). To use our dataloader, put the data split file `dataset/Imagette/train.csv (val.csv)`  under the data directory and organize the folder as
```
imagenette2/
	└── train/ 
	└── val/
	└── train.csv
	└── val.csv
```
### CUB-GHA
The data can be downloaded from the [official page](https://github.com/yaorong0921/CUB-GHA/tree/main). To use our dataloader, put the data split file `dataset/CUB-GHA/train.csv (test.csv)`  under the data directory and organize the folder as

```
CUB_200_2011/
	└── images/ 
	└── CUB_GHA/
	└── train.csv
	└── test.csv
```

## Training

### Training on Imagenette
```
python imagenet_train.py --mode elastic --training fast --epsilon 0.01 --epsilon2 0.05 --data_dir ./imagenette2 --snapshot_path checkpoint
```
`--mode`: the variants of adversarial training, can be one of `standard`, `l1`, `elastic`, `group`.

`--training`: to use ietrative optimization or one-step optimization, can be `fast` or `iterative`.

`--epsilon`: the coefficient $\epsilon$ in l1 norm and group norm mode, or $\epsilon_1$ in elastic net mode.

`--epsilon2`: the coefficient $\epsilon_2$ in elastic net mode.

`--data_dir`: directory of the training data.

`--snapshot_path`: directory to store the trained model.



### Training on CUB-GHA
```
python cub_train.py --epsilon 0.5 --data_dir ./CUB_200_2011 --snapshot_path checkpoint_cub
```

## Interpretation
We use simple gradient for interpretation. 

```python
from utils import simplegrad, agg_clip

saliency_map = simplegrad(net, x, label)
saliency_map = agg_clip(saliency_map)
plt.imshow(saliency_map)
```
where `net` is the trained neural network, `x` is an image of shape `[1,3,224,224]` and label is can be ground truth label or the predicted label.

## Bibtex
If you find this work helpful, you can cite our paper as follows:
```
@inproceedings{gong2024structured,
  title={Structured Gradient-based Interpretations via Norm-Regularized Adversarial Training},
  author={Gong, Shizhan and Dou, Qi and Farnia, Farzan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11009--11018},
  year={2024}
}
```
