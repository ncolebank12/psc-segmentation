# PSC-SEGMENTATION: Perovskite Solar Cell Segmentation with YOLOv11

This repository documents experimentation on training a YOLOv11 segmentation model to predict the edges of perovskite solar cell samples in an image.

## Repository Structure

```
├── datasets
│   ├── images         # Training and validation images
│   └── labels         # Corresponding segmentation labels
├── mykernel           # Custom kernel configurations
├── runs               # Training runs and output logs
├── .gitignore         # Git ignore file
├── data.yaml          # Dataset configuration
├── readme.md          # This file
├── requirements.txt   # Python dependencies
├── train.txt          # Training file paths
├── yolo_training.ipynb # Training notebook
├── yolo_training.py   # Python script for model training
```

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset consists of Perovskite solar cell images and their corresponding segmentation masks:
- `datasets/images`: Contains source images of Perovskite solar cells. Images are divided into `train` and `val` folders.
- `datasets/labels`: Contains segmentation labels in YOLO format (exported from cvat.ai). Labels are divided into `train` and `val` folders.

Dataset configuration is specified in `data.yaml`.

## Training

You can train the model using either the Jupyter notebook or Python script:

### Using Jupyter Notebook

Open and run `yolo_training.ipynb` which provides a step-by-step process, best for testing different configurations

### Using Python Script

```bash
python yolo_training.py
```

This is best for submitting a non-interactive job to a node (e.g. DSMLP)

## Image Labeling
All image labeling was done using the platform [cvat](cvat.ai). It allows for segmentation labeling and exporting annotations directly into YOLO format.

## runs/segment Directory

Training results and performance metrics are stored in the `runs` directory, including:
- Confusion matrices
- Precision-recall curves
- Sample predictions (in the folders starting with "predict_")
- Training logs

The `runs` directory has folders corresponding to the training results and the predictions made on the validation set.

For example, the folder `m_1280_500e` corresponds to training results of a model with size m, an image size of 1280, and a training time of 500 epochs. The `predict_m_1280_500e` folder contains the predicted masks for the model trained in `m_1280_500e`

## Results

Training started with 68 annotations, 60 of which are for training and 8 for validation. The first model tested was with model size n, image size 640, and run for 100 epochs.

The training graphs for this are below:

![n_640_100e training graphs](https://github.com/ncolebank12/psc-segmentation/blob/main/runs/segment/n_640_100e/results.jpg?raw=true)

I then experimented with different image sizes (640 vs 1280) model size (m vs n) and increasing epochs to 500. The training results for these can be found in the corresponding folders within the `runs/segment` directory.

Lastly, I implemented the following augmentations to the training, which effectively creates more useful training data from the preexisting images:

`augment:
  hsv_h: 0.015  # Hue adjustment
  hsv_s: 0.7    # Saturation adjustment
  hsv_v: 0.4    # Value adjustment
  flipud: 0.5   # Vertical flip probability
  fliplr: 0.5   # Horizontal flip probability
  mosaic: 1.0   # Mosaic augmentation probability
  mixup: 0.2    # MixUp augmentation probability
  translate: 0.2
  scale: 0.5`

These are the results for the latest model (m_1280_500e_augmented):

![m_1280_500e_augmented training graphs](https://github.com/ncolebank12/psc-segmentation/blob/main/runs/segment/m_1280_500e_augmented/results.jpg?raw=true) 

Here are some of the predicted segmentations from the latest model:

![m_1280_500e_augmented prediction 1](https://github.com/ncolebank12/psc-segmentation/blob/main/runs/segment/predict_m_1280_500e_augmented/G0010720.jpg?raw=true) 

![m_1280_500e_augmented prediction 2](https://github.com/ncolebank12/psc-segmentation/blob/main/runs/segment/predict_m_1280_500e_augmented/G0013270.jpg?raw=true) 

![m_1280_500e_augmented prediction 3](https://github.com/ncolebank12/psc-segmentation/blob/main/runs/segment/predict_m_1280_500e_augmented/G0048338.jpg?raw=true) 