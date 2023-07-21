# Spatial Structure Constraints for Weakly Supervised Semantic Segmentation


Introduction
------------
This is the source code for our paper **Spatial Structure Constraints for Weakly Supervised Semantic Segmentation**

Network Architecture
--------------------
The architecture of our proposed approach is as follows
![network](framework.png)

## Installation

* Install PyTorch 1.7 with Python 3 and CUDA 11.3

* Clone this repo
```
git clone https://github.com/NUST-Machine-Intelligence-Laboratory/SSC.git
```

### Download PASCAL VOC 2012 

* Download [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
* Download [Superpixel](https://wsss-ssc.oss-cn-shanghai.aliyuncs.com/voc_superpixels.zip)



## Training

* Run run_sample.py (You can either mannually edit the file, or specify commandline arguments.) 
```
python run_sample.py
```

