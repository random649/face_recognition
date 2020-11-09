### environment

pytorch == 1.2.0

torchvision == 0.4.0

opencv-python == 4.4.0.44



### data

Selected 4000 person (25 images for each)  from UmdFaces dataset. 

[UmdFaces](https://pan.baidu.com/s/1tHdNy-AO6REd2ADTJRI_-A) 提取码：yc9p

[lfw](https://pan.baidu.com/s/1wPbZ5_-GCNRrKs0D0QiKWA) 提取码：mm1v

[lfw_test_pair](https://pan.baidu.com/s/1BgyZsLNrM1PFFoWB_VzYxg) 提取码：vjnu



### run

1、修改train.py中parser的参数

2、python train.py

3、修改test.py中parser的参数

4、python test.py



目前可以调整的参数：

Backbone: num_layers=50/100/152  mode='ir'/'ir_se'

train: batch_size  num_epoch  lr  stages  weight_decay