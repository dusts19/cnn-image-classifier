## CINIC-10 Image Classification with Deep CNN and K-Fold Cross-Validation

This project implements a deep convolutional neural network in PyTorch to classify images from the [CINIC-10 dataset](https://datashare.ed.ac.uk/handle/10283/3192), a challenging benchmark dataset that bridges CIFAR-10 and ImageNet subsets.

## Features
 - Custom CNN with residual-style skip connections
 - Trained on **CINIC-10** using **7-fold cross-validation**
 - Data augmentation: 'RandomHorizontalFlip', 'RandomRotation'
 - Early stopping to prevent overfitting
 - ROC curve + AUROC visualizations per class
 - Model summary using 'torchsummary'
 - Visualization of sample and batch images
 - Tracks loss/accuracy over epochs and folds

## Model Architecture
 - 6 Convolutional layers with BatchNorm, ReLU, and Dropout
 - 3 Residual (skip) connections using 1x1 convolutional shortcuts
 - 2 Fully connected layers (128 hidden units)
 - Output layer for 10-class classification (softmax over 100 for future extensibility)

Input Size: 32x32 RGB images
Total Parameters: ~8.6 million
     
     > **Note:** (_The final layer uses 'LogSoftmax' activation; note that this should be paired with 'NLLLoss', or the layer should be removed when using 'CrossEntropyLoss' - see Notes._)

## Requirements:
- Python 3.8+
- Anaconda (or 'venv')
- Required libraries (install with pip)
  ```bash
  pip install torch torchvision matplotlib scikit-learn torchsummary torchviz tqdm kaggle
  ```
  
## Dataset
Dataset: [mengcius/cinic10](https://www.kaggle.com/datasets/mengcius/cinic10) on Kaggle.

## Set up Kaggle Credentials
To access the dataset programatically, you'll need a Kaggle API key (kaggle.json)
 1. Go to kaggle.com -> Go to you Kaggle account -> **"Account"** -> **"Create API Token"**
 2. This downloads 'kaggle.json'
 3. Use one of the following options to make it available to your notebook - upload 'kaggle.json' to your working directory or Google Drive (if using Colab):
    #### Option A: Upload directly in Colab
      ```python
      from google.colab import files
      files.upload() # Upload kaggle.json
      ```
    #### Option B: Load from Google Drive in Colab
      ```python
      from google.colab import drive
      drive.mount('/content/drive')

      !mkdir -p ~/.kaggle
      !cp /content/drive/MyDrive/path/to/kaggle.json ~/.kaggle/
      !chmod 600 ~/.kaggle/kaggle.json
      ```
    #### Option C: Manual Environment Setup (Local or Jupyter)
      ```python
      import os
      os.environ['KAGGLE_USERNAME'] = 'your_username'
      os.environ['KAGGLE_KEY'] = 'your_api_key'
    Once kaggle.json is in place, the notebook handles dataset download automatically.



## Installation & Setup

### Option 1: Run on Google Colab
1. Open the provided notebook ('CNN_image_classifier_CINIC-10.ipynb') in Google Colab
2. Upload your 'kaggle.json' (from step 3 under **Dataset**) or mount your Google Drive
3. Run all cells to:
   - Install dependencies
   - Download and extract the dataset
   - Train the model with 7-fold cross-validation
   - Visualize results

### Option 2: Run Locally (Python ≥ 3.8, pip, and GPU recommended)
1. Clone the repo:
```bash
git clone [https://github.com/dusts19/cnn-image-classifier.git](https://github.com/dusts19/cnn-image-classifier)
cd cnn-image-classifier
```
2. Install the dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn torchsummary torchviz tqdm kaggle
```
3. Place kaggle.json in the project root directory and run the notebook or Python script.

## Evaluation
The notebook tracks and outputs the following:
 - Training & validation accuracy/loss per epoch
 - ROC Curves and AUROC per class
 - Final test set accuracy
 - Visualizations for sample predictions with true labels and confidence

<img width="821" alt="image" src="https://github.com/user-attachments/assets/c21c9a9e-a0e2-46b4-b757-f25354fc0380" />
<img width="664" alt="image" src="https://github.com/user-attachments/assets/a0dba5b2-c4c0-4458-b3ac-e836983332b9" />

## Notes
- The model uses 100 output units for potential extensibility beyond CINIC-10 (which has 10 classes).
- The final layer is currently LogSoftmax, which is not standard when using CrossEntropyLoss. For correctness, either:
  - Remove LogSoftmax and replace with raw logits, or
  - Change the loss function to NLLLoss
- Training with 7-fold cross-validation takes 2~3 hours on a Colab T4 GPU.

## Acknowledgements
- CINIC-10: CINIC-10 Dataset
- Kaggle dataset mirror by mengcius

## To-Do /Future Work
- Replace LogSoftmax + CrossEntropyLoss mismatch
- Add test-time augmentation
- Implement Grad-CAM visualizations
