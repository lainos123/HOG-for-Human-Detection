# HOG-based Human Detection System

This repository contains a human detection system using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier, based on the method introduced by Dalal and Triggs (CVPR 2005).

## Overview

We've developed a system that extracts HOG features from images and uses an SVM to classify them as either human or non-human. The project implements a complete pipeline from data preparation to feature extraction, model training, and evaluation through a graphical user interface.

## Setup
> ❗ **If you don’t have Conda installed**, download and install **Miniconda** (recommended) from:  
> https://docs.conda.io/en/latest/miniconda.html  
> This gives you access to `conda` without the full Anaconda package.

Once Conda is installed:

```bash
# Create the Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate cits4402_project_env
```

## Running the GUI

```bash
# Navigate to the Submission folder
cd Submission

# Run the GUI
python GUI.py
```
##### GUI Features:
The GUI allows you to test our model on any image. You can:
- Select images from the included "Testing Images" folder
- Upload your own images (64x128)
- View real-time classification results
- Save test results to .xlsx
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
  - 250 human samples from PETA MIT dataset
  - 250 non-human samples extracted from INRIA dataset

The model achieves:
- 99.5% accuracy on the perfect_200 test set
- 87.75% accuracy on the unperfect_200 test set

## Documentation

- Detailed analysis and results can be found in `Submission/Report.ipynb`
- Dataset descriptions are available in `docs/DATA_OVERVIEW.md`
- Model naming conventions are explained in `docs/MODEL_NAME_CONVENTION.md`
- Development notebooks and analysis can be found in the `notebooks/` directory:
  - `ablation.ipynb`: HOG parameter ablation studies
  - `test_all_models.ipynb`: Model evaluation and selection
  - `archive/`: Historical development notebooks including initial HOG and SVM implementations
    - `create_negatives_INRIA.ipynb`: Design of non-human image extraction from INRIA dataset

## Note

Only the `Submission` folder was included in the final project submission. It contains a standalone version of the application with the final model and sample images for testing.

## Generating Features and Models

Due to storage limitations, the `Features/` and `Models/` directories are empty in this repository. To generate your own features and models, follow these steps:

1. **Generate HOG Features**:
   ```bash
   cd Submission/Others/Scripts
   python extract_HOG.py
   ```
   This will prompt you to:
   - Select a dataset to process
   - Choose HOG parameters
   - Optionally save as final model features

2. **Train SVM Models**:
   ```bash
   python train_SVM.py
   ```
   This will:
   - Use the features generated in step 1
   - Train an SVM classifier
   - Save the model with parameters in the filename

3. **Evaluate Models**:
   ```bash
   python evaluate_any_model.py
   ```
   This allows you to:
   - Test any trained model
   - Evaluate performance on different test sets
   - View detailed metrics and visualizations

4. **Run Ablation Study**:
   ```bash
   python ablation.py
   ```
   This will:
   - Test different HOG parameter combinations
   - Compare performance across test sets
   - Help identify optimal parameters

## Repository Structure
```
├── Datasets/
│   ├── Raw/ # Source data
│   │   ├── original/ # INRIA and PETA source datasets
│   │   └── created/ # Processed subsets used for for test and train sets
│   ├── Test/ # Testing datasets of varying quality and size
│   └── Train/ # Training datasets with different compositions
├── Features/ # Directory for pre-extracted HOG features (.npy files) - empty, generate using extract_HOG.py
├── Models/ # Directory for trained SVM classifier models (.joblib files) - empty, generate using train_SVM.py
├── Submission/ # Final submission folder with standalone application
│   ├── GUI.py # Graphical user interface
│   ├── Others/ # Supporting files
│   │   └── Scripts/ # Core functionality scripts
│   │       ├── ablation.py # HOG parameter ablation study script
│   │       ├── create_non_human_data.py # INRIA_non_human_all dataset creation
│   │       ├── create_test_data.py # Test dataset creation
│   │       ├── create_train_data.py # Training dataset creation
│   │       ├── evaluate_any_model.py # Testing an individaul model of choice
│   │       ├── evaluate_final_model.py # Testing the final model
│   │       ├── extract_HOG.py # HOG feature extraction
│   │       ├── save_predictions.py # Save prediction results from GUI
│   │       ├── test_image.py # Single image testing
│   │       └── train_SVM.py # SVM model training
│   ├── Testing Images/ # Sample images for GUI testing
│   └── Report.ipynb # Submitted Report
├── docs/
│   ├── DATA_OVERVIEW.md # Description of dataset organisation
│   └── MODEL_NAME_CONVENTION.md # Explanation of model naming scheme
├── notebooks/
│   ├── ablation.ipynb # HOG parameter ablation studies - explanation of the script
│   ├── archive/ # Historical development notebooks
│   │   ├── 00019_male_fore.jpg # image used for testing
│   │   ├── Franco_HOG_SVM.ipynb # Franco's version of initial implementaiton of HOG  extraction and SVM training
│   │   ├── create_negatives_INRIA.ipynb # Design of non_human extraction from INRIA
│   │   └── phase_2_laine.ipynb # Laine's version of initial implementaiton of HOG extraction and SVM training
│   ├── outputs/ # Generated analysis plots and results
│   └── test_all_models.ipynb # Model evaluation - finding the best training set to uses for Final Model
└── environment.yml # Conda environment configuration
```

## Contributors

- **Laine Mulvay** (Student ID: 22708032)
- **Franco Meng** (Student ID: 23370209)

