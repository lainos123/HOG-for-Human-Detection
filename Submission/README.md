# HOG-based Human Detection System

This repository contains the submission materials for our human detection system using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier, based on the method introduced by Dalal and Triggs (CVPR 2005).

## Overview

We've developed a system that extracts HOG features from images and uses an SVM to classify them as either human or non-human. This submission folder contains the standalone application with GUI interface, testing images, and detailed report.

## Setup

> ❗ **If you don't have Conda installed**, download and install **Miniconda** (recommended) from:  
> https://docs.conda.io/en/latest/miniconda.html  
> This gives you access to `conda` without the full Anaconda package.

Once Conda is installed:

```bash
# Create the Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate cits4402_project_submission_env

# Run the GUI
python GUI.py
```

<div align="center">
  <img src="Others/example_images/GUI.png" alt="GUI in action" width="50%">
</div>

## GUI Features

The GUI allows you to test our model on various images:
- Select images from the included "Testing Images" folder
- Upload your own images (64x128 pixels)
- View real-time classification results
- Save test results to predictions.xlsx
- See confidence scores for each classification
- Visualise the HOG features extracted from the image

## Final Model

Our final model uses the following parameters:
- **HOG parameters**: 
  - Cell Size: 8×8 pixels
  - Block Size: 16×16 pixels (2×2 cells)
  - Orientation Bins: 9 bins (0°-180°)
  - Block Stride: 1 cell (8 pixels)
  - Filter: Default gradient filter
  - Normalisation: L2-Hys

- **SVM parameters**: 
  - Linear kernel
  - C=1.0
  - Probability estimates enabled

- **Training data**: 
  - 250 human samples from PETA MIT dataset, consisting of high-quality human images
  - 250 non-human samples extracted from INRIA dataset, carefully selected to contain no humans

The model achieves:
- 99.5% accuracy on the perfect_200 test set
- 87.75% accuracy on the unperfect_200 test set

## Model Development

Our ablation study explored various HOG parameter combinations, including different cell sizes, block sizes and orientation bins. The study revealed an interesting trade-off: while some parameter combinations showed improved performance on our diverse training dataset, the optimal model performed worse on our final, simpler test cases. This suggests that the model was overfitting to the complex variations in the training data, leading to poorer generalisation. We found that the default parameters (8×8 cell size, 16×16 block size, 9 bins covering 180 degrees) provided the best balance, maintaining strong performance across both simple and challenging test scenarios. This finding highlights the importance of carefully balancing model complexity with generalisation capability.

## Documentation

- Detailed analysis and results can be found in `Report.ipynb`
- Dataset descriptions are available in `Others/docs/DATA_OVERVIEW.md`
- Model naming conventions are explained in `Others/docs/MODEL_NAME_CONVENTION.md`
- Development notebooks and analysis can be found in the `Others/notebooks/` directory:
  - `ablation.ipynb`: HOG parameter ablation studies
  - `test_all_models.ipynb`: Model evaluation and selection
  - `archive/`: Historical development notebooks including initial HOG and SVM implementations
    - `create_negatives_INRIA.ipynb`: Design of non-human image extraction from INRIA dataset

## Dataset Sources

Our datasets were carefully constructed from two main sources:

### Raw Data Sources
- **PETA MIT Dataset**: A subset of the PEdesTrian Attribute (PETA) dataset containing 2,023 high-quality human images.
- **INRIA Person Dataset**: Contains 1,811 images with XML annotations marking human positions, used as a source for both human and non-human samples.

### Dataset Creation Process
- **Human Samples**: Selected from the PETA MIT dataset, ensuring clear, high-quality images of people.
- **Non-Human Samples**: Created using our `create_non_human_data.py` script, which:
  - Processes the INRIA dataset and its XML annotations
  - Extracts regions without humans by checking areas to the left and right of human annotations
  - Scales these regions to 64x128 pixels (the standard size for our model)
  - Saves them as negative samples

### Testing Datasets
- **perfect_200**: 200 high-quality test samples
  - 200 human samples from PETA
  - 200 non-human samples from carefully selected INRIA regions with no humans
- **unperfect_200**: 200 test samples including some challenging examples
  - 150 human samples from PETA + 50 from INRIA with partial or multiple humans
  - 150 non-human samples from INRIA + 50 patches from images with no humans

All datasets were created using a fixed random seed (4402) for reproducibility, and we ensured no overlap between training and testing sets.

## Custom Pipeline: Feature Extraction, Training, and Evaluation

Easily build and test your own pipeline using the provided scripts:

1. **Extract HOG Features:**
  - Run `extract_HOG.py` to extract features from any dataset with custom parameters.
  - Features are saved in `Features/`.

  ![Extract](Others/example_images/extract_custom_features.png)

2. **Train SVM Model:**
  - Run `train_SVM.py` to train on any set of features.
  - Models are saved in `Models/`.

  ![Train](Others/example_images/train_custom_model.png)

3. **Evaluate Model:**
  - Run `evaluate_any_model.py` to test any model on any test set.
  - View results and metrics interactively.
  
   ![Evaluate](Others/example_images/evaluate_any_model.png)

This modular workflow lets you experiment with any combination of datasets, features, and models.

## Submission Structure

The submission folder contains the following structure:
```
.
├── GUI.py                      # Main application with GUI interface for testing the model
├── Others/
│   ├── Final Model/            # Contains the trained SVM model with optimal parameters
│   │   └── svm_hog_classifier_PETA_INRIA_h250p_nh250pp_c4_b32_n9_s1_180.joblib
│   ├── Scripts/                # Core functionality scripts for model development and testing
│   │   ├── create_non_human_data.py    # Creates non-human dataset from INRIA by extracting regions without humans
│   │   ├── create_test_data.py         # Generates test datasets with varying quality and composition
│   │   ├── create_train_data.py        # Prepares training datasets with different sample compositions
│   │   ├── evaluate_any_model.py       # Evaluates performance of any trained model
│   │   ├── extract_HOG.py              # Extracts HOG features from images using specified parameters
│   │   ├── save_predictions.py         # Saves classification results from GUI to predictions.xlsx
│   │   ├── test_image.py               # Tests single image classification with confidence scores
│   │   └── train_SVM.py                # Trains SVM classifier with specified parameters
│   ├── docs/                   # Documentation files explaining project organisation
│   │   ├── DATA_OVERVIEW.md            # Detailed description of dataset organisation and sources
│   │   ├── MODEL_NAME_CONVENTION.md    # Explanation of model naming scheme and parameters
│   │   └── Report.md                   # Comprehensive project report
│   ├── example_images/         # Example images used in report
│   └── notebooks/             # Development notebooks for analysis and experimentation
│       ├── ablation.ipynb              # HOG parameter ablation studies - detailed analysis of parameter combinations
│       ├── archive/                    # Historical development notebooks
│       │   ├── Franco_HOG_SVM.ipynb    # Initial HOG and SVM implementation by Franco
│       │   ├── create_negatives_INRIA.ipynb  # Design of non-human extraction methodology
│       │   └── phase_2_laine.ipynb     # Initial HOG and SVM implementation by Laine
│       ├── outputs/                    # Generated analysis plots and results
│       └── test_all_models.ipynb       # Model evaluation - finding optimal training set composition
├── README.md                  # This file
├── Report.pdf                 # Final project report with detailed analysis
├── Testing Images/            # Sample images for testing the GUI interface
└── environment.yml            # Conda environment configuration with required dependencies

```

## Full Repository

The complete project repository includes additional resources:
- Various training and testing datasets
- Pre-extracted HOG features
- Multiple trained SVM models
- Development notebooks and ablation studies
- Documentation on dataset organisation and model naming conventions

To access the full repository:
```bash
git clone https://github.com/lainos123/HOG-for-Human-Detection
```

Visit [https://github.com/lainos123/HOG-for-Human-Detection](https://github.com/lainos123/HOG-for-Human-Detection) for more details and the comprehensive README.

## Contributors

- **Laine Mulvay** (Student ID: 22708032)
- **Franco Meng** (Student ID: 23370209)