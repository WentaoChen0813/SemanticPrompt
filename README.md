## Semantic Prompt for Few-Shot Learning

This is the PyTorch implementation of the proposed Semantic Prompt (SP) approach.

### Requirements
* Python >= 3.8
* PyTorch >= 1.7.1
* clip (https://github.com/openai/CLIP)
* sentence_transformers (https://github.com/UKPLab/sentence-transformers)

### Datasets
* miniImageNet: https://rec.ustc.edu.cn/share/1341cd00-ffa6-11ed-8581-c30591dc01d6
* tieredImageNet: https://rec.ustc.edu.cn/share/3df5e760-ffa6-11ed-accd-c197f7deb7f7
* CIFAR-FS: https://rec.ustc.edu.cn/share/58e3c480-ffa6-11ed-bdc6-31dddcd9f8de
* FC100: https://rec.ustc.edu.cn/share/72752780-ffa6-11ed-86c2-435b9749436f


Download the dataset you need and put the xxx.tar.gz in ./dataset
```
cd ./dataset
tar -xvzf xxx.tar.gz
```

### Scripts
#### Pre-train the feature extractor
* miniImageNet
```
python train_vit.py --gpu 0 --dataset miniImageNet --exp pre-train --rand_aug --repeat_aug
```
* tieredImageNet
```
python train_vit.py --gpu 0 --dataset tieredImageNet --exp pre-train --rand_aug --repeat_aug --epochs 300
```
* CIFAR-FS
```
python train_vit.py --gpu 0 --dataset CIFAR-FS --exp pre-train --rand_aug --repeat_aug
```
* FC100
```
python train_vit.py --gpu 0 --dataset FC100 --exp pre-train --rand_aug --repeat_aug
```

#### Fine-tune the model with SP
* miniImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp sp --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth
```

* tieredImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset tieredImageNet --exp sp  --rand_aug --train_episodes 600 --init checkpoint/tieredImageNet/visformer-t/pre-train/checkpoint_epoch_300.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset tieredImageNet --exp sp_5shot --shot 5  --rand_aug --train_episodes 600 --init checkpoint/tieredImageNet/visformer-t/pre-train/checkpoint_epoch_300.pth
```

* CIFAR-FS
```
1-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp sp --init checkpoint/CIFAR-FS/visformer-t/pre-train/checkpoint_epoch_800.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp sp_5shot --shot 5 --init checkpoint/CIFAR-FS/visformer-t/pre-train/checkpoint_epoch_800.pth
```
* FC100
```
1-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp sp --init checkpoint/FC100/visformer-t/pre-train/checkpoint_epoch_800.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp sp_5shot --shot 5 --init checkpoint/FC100/visformer-t/pre-train/checkpoint_epoch_800.pth
```

#### Test
* miniImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test --test --episodes 2000 --resume checkpoint/miniImageNet/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/miniImageNet/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```

* tieredImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset tieredImageNet --exp test --test --episodes 2000 --resume checkpoint/tieredImageNet/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/tieredImageNet/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```

* CIFAR-FS
```
1-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp test --test --episodes 2000 --resume checkpoint/CIFAR-FS/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/CIFAR-FS/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```

* FC100
```
1-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp test --test --episodes 2000 --resume checkpoint/FC100/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/FC100/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```