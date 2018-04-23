# autocolorization

based on Richard Zhang's paper:([Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf))

and his source code: ([Richard Zhang's Github](https://github.com/richzhang/colorization))

#### Requirment

```
TensorFlow etc. 
```


#### Train

## 1. link your training dataset directory to the following path: 

```
ln -s $your_data_path data/imagenet
```
  
## 2. create the list of paths of training data:

```
python tools/create_imagenet_list.py
```

## 3. train (training setting can be changed in "train.cfg" file)

```
python tools/train.py -c conf/train.cfg
```

#### Test/demo

## 1. Download pretrained model and move it to models/
[pretrained model](https://drive.google.com/file/d/0B-yiAeTLLamRWVVDQ1VmZ3BxWG8/view?usp=sharing)

## 2. Test/demo

```
python demo.py
```
