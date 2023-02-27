## Implementation of VRL4CCD in Pytorch
---

## Attention
**Datasets and training your own datasets can be in two different formats**. Need to pay attention to the placement of the format！  

The warehouse implements the Siamese network, which is often used to detect the similarity of two input code images. The backbone feature extraction network (backbone) used in this warehouse is VGG16.  

## Environment
Configure the environment according to requirements.txt.
```
pip install -r requirements.txt
```

## Download
Need to download vgg16-397923af.pth pre-training weights    
Link: https://pan.baidu.com/s/14SFoKX6xTDPx2XG9rcUTDQ Extraction code: 44en       

## Visualization
1. Download clone fragments from https://github.com/clonebench/BigCloneBench
2. Modify the file path in **codeVis.py**.
1. Run **codeVis.py** to generate code images.

## How2predict
### Use your own trained weights
1. Follow training steps. 
2. Modify the **model_path** in the **siamese.py** file to correspond to the trained file. The trained file is in the **logs** folder.
  
```python
_defaults = {
    "model_path": 'model_data/vgg.pth',
}
```

3. Run predict.py, enter  
```python
your_img/xxx.png
```

## How2train  
### Train your own model for clone detection.
If you want to train your own model, you can arrange the data set in the following format. For example:
```python
- images_background
	- character01
		- 0709_01.png
		- 0709_02.png
		- ……
	- character02
	- character03
	- ……
```
    
The training steps are:
1. Place the dataset according to the above format, and put it in the dataset folder.
2. Then set the **train_own_data** in **train.py** to True.
3. Run **train.py** to start training. 

### Reference
You can see more descriptions of **Siamese Neural Networks** in the: https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning
