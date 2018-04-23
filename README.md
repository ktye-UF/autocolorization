# autocolorization

based on Richard Zhang's paper:([Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf))

### Train

## link your training dataset directory to the following path:

	```
	ln -s $Imagenet data/imagenet
	```
  
## create the list of paths of training data:

```
python tools/create_imagenet_list.py
```

## train (training setting can be changed in "train.cfg" file)

```
python tools/train.py -c conf/train.cfg
```

### Test/demo

1. Download pretrained model and move it to models/
(<a>https://drive.google.com/file/d/0B-yiAeTLLamRWVVDQ1VmZ3BxWG8/view?usp=sharing</a>)

2. Test/demo

	```
	python demo.py
	```
