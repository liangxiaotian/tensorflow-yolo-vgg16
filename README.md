# tensorflow-yolo

 Use vgg16 pretrian model to train YOLO
 
 tensorflow 1.4.0
 python 3.6.0
 
 ## DataSet：
 
 ln -s $your_data data/voc
 
 python preprocess_pascal_voc.py  
 
 
 ## Train:
 python tools/train.py
