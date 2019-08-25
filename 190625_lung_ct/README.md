# tianchi competition
对肺结节、索条、动脉硬化或钙化、淋巴结钙化等病种进行定位和疾病分类  
https://tianchi.aliyun.com/competition/entrance/231724/information  
# reference
肺结节、索条、动脉硬化或钙化、淋巴结钙化四种症状简介：explain.docx  
切片视频：sample.mp4

Preprocess data and train unet tutorial:Data Science Bowl 2017  
https://www.kaggle.com/c/data-science-bowl-2017/overview/tutorial


## 相关资料
### 历届大赛资料：  
1 https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.12f4379cxM73f2&postId=2947  
2 https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.12f4379cxM73f2&postId=2966  
3 https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.12f4379cxM73f2&postId=2893  
4 https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.12f4379cxM73f2&postId=2915  
7 https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.12f4379cxM73f2&postId=2898  

### objection detection 资料
1 kaggle 比赛top solution开源整理（3个月前整理的（2019.03左右））里面有detection相关比赛的开源代码，我看到有第二的：  
https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions  
2 在coco数据集上的开源整理，现在[coco排行榜]( http://cocodataset.org/#detection-leaderboard )最好的到52分,是fece++的，不过我没找到代码，下面这个链接最好的精度在coco上是第三名：  
https://paperswithcode.com/sota/object-detection-on-coco  
3. **kaggle 一个医学图像的object detection的第一名（8个月前）**（[方法](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70421)和[代码](https://github.com/i-pan/kaggle-rsna18)）




## 注意
1. 本次数据标注不是球体，而是长方体，并且标出来的就是长方形的边长
2. diameterZ表示z轴的切片数，标注中有时算出来的z轴中心是0.5的小数，则一般z轴的切片是偶数，z轴左右各有数据


## 上传记录
1. 7-19提交：使用模型faster-rcnn 地址：https://github.com/jwyang/faster-rcnn.pytorch.git，使用学习率1e-4，预训练模型使用resnet101，数据集使用code/read_data/convert_dataset_to_voc2007_rgb.py转换,这个模型中rgb的效果似乎比灰度好那么一点，后跟据HU值生成出两个数据集，肺窗数据集（具体范围多少记不清了，在公司的电脑上，改天把数据集一并上传）和纵膈窗数据集，用肺窗数据集训练结节、索条，用纵膈窗数据集训练动脉硬化或钙化、淋巴结钙化，最后把两预测结果相加  
2. 7-23提交：完成了voc转coco格式的代码，这样可以使用mmdetection中的模型，具体使用方法参照mmdetection的[get start文档](https://github.com/open-mmlab/mmdetection/blob/master/GETTING_STARTED.md) 即可
3. 上传grid_1：日期:2019-07-27 16:21:00 score:0.3074 grid模型初始版本
4. 上传grid_2：val+train都拿来训，图片的均值和方差都改成数据集的均值和方差