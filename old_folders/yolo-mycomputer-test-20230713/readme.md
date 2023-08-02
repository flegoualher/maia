# How to use YOLO v8 ?

https://docs.ultralytics.com/

### 1. Data

#### 1.1 Folder structure

<img src="/Users/user/Desktop/Screenshot 2023-07-13 at 10.04.14 AM.png" alt="Screenshot 2023-07-13 at 10.04.14 AM" style="zoom:50%;" />

Images and masks should be placed in the ==datasets== folder with a dataset folder name (==images-v1==).

==images-v1== folder's structure is fixed, as well as the files names : ==tile_{number}.jpeg==.

Only copies of the images are placed here, not the original dataset.

Images have to be .jpeg files, placed in the ==images-v1== folders (separated into ==train== and ==val== folders)

Masks are placed in the ==labels== folder (separated into ==train== and ==val== folders). only .txt files.

#### 1.2 Folder name

If you change the images-v1 folder name, you have to update the custom.yaml file.

#### 1.3 Images

Choose the images you want for training and validation and place them in ==datasets/images-v1/train== and ==datasets/images-v1/val== folders (jpeg only).

The model_run.ipynb Jupyter Notebook Section 1.1 helps to create jpeg images and copy them in the correct folders, once input/output paths are adjusted.

#### 1.4 Masks

Masks are created using a two step-process :

- **step 1 : conversion of original tiff masks into .json file**

Just launch the terminal, choose the directory of the main_maia.py file and execute it by typing 

`python main_maia.py`

The .json file is created in the ==output-json== folder.

- **step 2 : conversion of .json file into .txt files**

Execute the cells in Section 1.2 of the model_run.ipynb Jupyter Notebook file. The one containing

`convert_coco_json(json_dir='../output-json/', use_segments=True, cls91to80=False)`

creates the .txt masks for each tile in the ==output-yolo== folder.

#### Verify the masks

After creating an account on roboflow.com, you can verify the masks using this address :

https://app.roboflow.com/maia/maia/upload#

Drag and drop the ==images-v1== folder from the ==datasets== folder. You should be able to see images with masks.



### 2. YOLO v8 training

The folder ==yolov8== contains the model_run.ipynb file. Section 3 :  YOLO v8 contains the code to launch training.

The size of the Neural Network can be adjusted by updating the letter **m** in yolov8m-seg.pt  

`model = YOLO('yolov8m-seg.pt')` 

n > s > m > l > x : n for nano (simplest model), x for xtra model (most complicated model)

The ==yolov8/runs== folder contains the results.



### 3. Metrics : mAP50

https://medium.com/axinc-ai/map-evaluation-metric-of-object-detection-model-dd20e2dc2472

##### About IOU

##### Object detection models predict the bounding box and category of objects in an image. *Intersection Over Union (IOU)* is used to determine if the bounding box was correctly predicted.

The IOU indicates how much bounding boxes overlap. This ratio of overlap between the regions of two bounding boxes becomes 1.0 in the case of an exact match and 0.0 if there is no overlap.

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*fYdiMfuzhqJ5OtSKmo_xtQ-20230713213005507.png)

Source: https://github.com/rafaelpadilla/Object-Detection-Metrics

In the evaluation of object detection models, it is necessary to define how much overlap of bounding boxes with respect to the ground truth data should be considered as successful recognition. For this purpose, IOUs are used, and *mAP50* is the accuracy when IOU=50, i.e., if there is more than 50% overlap, the detection is considered successful. The larger the IOU, the more accurate the bounding box needs to be detected and the more difficult it becomes. For example, the value of *mAP75* is lower than the value of *mAP50*.

# About Precision and Recall

*Precision* is the ability of a model to identify only the relevant objects. It answers the question *What proportion of positive identifications was actually correct*? A model that produces no false positives has a precision of 1.0. However, the value will be 1.0 even if there are undetected or not detected bounding boxes that should be detected.

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*uZk9UCjL6JWYXWmpZ23Log.gif)

Source: https://github.com/rafaelpadilla/Object-Detection-Metrics

*Recall* is the ability of a model to find all ground truth bounding boxes. It answers the question *What proportion of actual positives was identified correctly?* A model that produces no false negatives (i.e. there are no undetected bounding boxes that should be detected) has a recall of 1.0. However, even if there is an “overdetection” and wrong bounding box are detected, the recall will still be 1.0.

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*CQSLw7zTiDyjleV3J5vkSA.gif)

Source: https://github.com/rafaelpadilla/Object-Detection-Metrics

# About Precision Recall Curve

The *Precision Recall Curve* is a plot of *Precision* on the vertical axis and *Recall* on the horizontal axis.

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*hKq9q4TJ7BSDxg6TrGHTRA.png)

Source: https://github.com/rafaelpadilla/Object-Detection-Metrics

There is a threshold for object detection. Increasing the threshold reduces the of risk of over-detecting objects, but increases the risk of missed detections. For example, if threshold=1.0, no object will be detected, *Precision* will be 1.0, and *Recall* will be 0.0. On the other hand, if threshold=0.0, an infinite number of objects will be detected, Precision will be 0.0, and Recall will be 1.0. Conversely, if threshold=0.0, an infinite number of objects will be detected, *Precision* will be 0.0, and *Recall* will be 1.0.

In the case of a good machine learning model, over-detection will not occur even if threshold is reduced (*Recall* is increased), and *Precision* will remain high. Therefore, the higher up the curve to the right in the graph, the better the machine learning model is.

# About AP

When comparing the performance of two machine learning models, the higher the *Precision Recall Curve*, the better the performance. It is time-consuming to actually plot this curve, and as the *Precision Recall Curve* is often zigzagging, it is subjective judgment whether the model is good or not.

A more intuitive way to evaluate models is the *AP (Average Precision)*, which represents the area under the curve (AUC) *Precision Recall Curve*. The higher the curve is in the upper right corner, the larger the area, so the higher the *AP*, and the better the machine learning model.

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*uNRonqOivovAPwspbAp0sQ.png)

Source: https://github.com/rafaelpadilla/Object-Detection-Metrics

# About mAP

The *mAP* is an average of the *AP* values, which is a further average of the *AP*s for all classes.

# Maximizing mAP

The *mAP* is calculated by fixing the *confidence threshold*. *COCO2017 TestSet* can be used to measure *mAP* on various *confidence thresholds* to check the effect of this threshold.

As a result, we confirmed that the smaller the *confidence threshold* is, the higher the *mAP* becomes.

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*OmqjV2xVCoKKsOYZW8j4Kg.png)

mAP50 for various thresholds measured on yolov4-tiny and yolov3-tiny

![img](/Users/user/code/flegoualher/maia/yolo-mycomputer-test-20230713/1*cDUWhnHw6sKjeq943xhkzw.png)

mAP75 for various thresholds measured on yolov4-tiny and yolov3-tiny

This result suggests that the more over-detection occurs, the higher the *mAP. A* higher *Recall* will result in a larger area than a higher *Precision*, and we believe this is due to the small number images (40 670) in *COCO2017 TestSet*.

In the script `test.py` of the yolov5 repository, the *confidence threshold* for *mAP* calculation has an extremely small value of 0.001.