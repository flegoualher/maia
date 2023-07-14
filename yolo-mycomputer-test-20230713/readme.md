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