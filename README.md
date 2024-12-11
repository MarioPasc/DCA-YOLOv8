# DCA-YOLOv8

> This is an improved framework for the automated detection of cardiac coronary artery stenosis based on the Yolov8 model, which has functions such as stenosis classification recognition and object detection.

## environment

1. Python3.7+

2. Pytorch1.1+
3. Conda11+
4. Yolov8

## start

1. Download dataset

2. Data preprocessing

   ```python
   python Canny_edge_extraction.py
   python Histogram_Equalization.py
   python Image_Fusion.py
   ```

3. Result to`Dataset`fold

4. Run detect.py

   ```python
   python detect.py
   ```

## dataset

1. Stenosis classification dataset:：https://github.com/KarolAntczak/DeepStenosisDetection/tree/master/

2. Stenosis target detection dataset：[The data underlying the results presented in this study are available in Dataset II, [10].](https://data.mendeley.com/datasets/p9bpx9ctcv/2)