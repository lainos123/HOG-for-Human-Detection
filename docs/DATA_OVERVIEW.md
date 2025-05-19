# Overview of Datasets used

## File Structure

The datasets are organised into three main directories:

1. **Raw**: Contains original and processed data source files
   - **original**: Direct downloads from source datasets (INRIA and PETA)
   - **created**: Processed data subsets created through different methods:
     - INRIA_non_human_all: Generated using `/Submission/Others/Scripts/create_non_human_data.py` (see `/notebooks/create_negatives_INRIA` for logic, though no longer runnable)
     - Other INRIA non-human sets: Manually selected subsets from INRIA_non_human_all
     - patches_non_human: Extracted from images without humans using patch extraction

2. **Train**: Contains different training dataset configurations created using `/Submission/Others/Scripts/create_train_data.py`
   - Each subdirectory represents a specific training dataset composition
   - Naming convention describes the mix of human/non-human samples
   - Each training set has separate human_train and non_human_train folders
   - Images are randomly selected from source datasets while ensuring no overlap with test sets

3. **Test**: Contains testing datasets of varying quality, created using `/Submission/Others/Scripts/create_test_data.py`
   - **perfect_100**: Small test set with clear human/non-human samples (100 samples)
   - **perfect_200**: Medium test set with clear human/non-human samples (200 samples)
   - **unperfect_200**: Medium test set with more ambiguous samples (200 samples)
   - Images are randomly selected from source datasets with a fixed random seed (4402) for reproducibility

## Datasets

### 1. Raw Data

The raw data is organised into two subfolders:

#### Original
Contains the unmodified source datasets:
- **PETA_2023**: The MIT subset of the [PEdesTrian Attribute (PETA) dataset](https://www.dropbox.com/scl/fi/35o74ndao1aofxodeytmk/PETA.zip?dl=0&e=2&file_subpath=%2FPETA+dataset%2FMIT%2Farchive&rlkey=37e4a29kdedcobjn5ojjcdixu), containing 2,023 images. See the [full PETA dataset](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) for more details.
- **INRIAPerson_1811**: The INRIA Person Dataset, containing 1,811 images with XML annotations marking human positions. This dataset was used as a source for generating non-human training samples by extracting regions without humans. Available at [Kaggle](https://www.kaggle.com/datasets/jcoral02/inriaperson).

#### Created
Contains processed subsets derived from the original datasets:
- **INRIA_non_human_all_1783**: Complete set of 1,783 non-human samples from INRIA, created using `/Submission/Others/Scripts/create_non_human_data.py`. This script processes the original INRIA dataset and its XML annotations to extract non-human regions by checking areas to the left and right of each human annotation. If no humans are detected in these regions, they are scaled to 64x128 pixels and saved as negative samples.
   - **INRIA_human_unperfect_301**: 301 hand-selected images from INRIA_non_human_all that contain either partial human figures or multiple humans
   - **INRIA_non_human_plus_1324**: Extended set of 1,324 non-human samples that may contain partial human figures or ambiguous cases where a human observer would not clearly identify a person in the image
      - **INRIA_non_human_perfect_501**: 501 images that contain no humans or parts of humans
- **patches_non_human**: Non-human image patches extracted from larger images

### 2. Training Datasets

Various training configurations with different mixtures of human/non-human samples. Images are randomly selected from source datasets while ensuring no overlap with test sets to maintain data independence.

### Naming Convention
The dataset names follow this format: `SOURCE_h[COUNT][TYPE]_nh[COUNT][TYPE]` where:
- `SOURCE`: Dataset source (PETA, INRIA, or both)
- `h`: Human samples
- `nh`: Non-human samples
- `[COUNT]`: Number of samples
- `[TYPE]`: Sample quality type
  - `p`: Perfect (clear, high-quality samples)
  - `u`: Unperfect (ambiguous or lower quality)
  - `pp`: Perfect+ (extended set that may include some ambiguous cases)
  - `pat`: Patches (extracted from larger images)
- `:`: Separator indicating different sources/types within the same category

### Training Configurations
- **PETA_INRIA_h250p_nh250pp**: 250 perfect human samples with 250 perfect+ non-human samples
- **PETA_INRIA_h250u:h250p_nh500pp**: Mix of 250 unperfect and 250 perfect human samples with 500 perfect+ non-human samples
- **PETA_INRIA_h500p_nh500pp**: 500 perfect human samples with 500 perfect+ non-human samples
- **PETA_INRIA_h800p:h200u_nh1000pp**: 800 perfect and 200 unperfect human samples with 1000 perfect+ non-human samples
- **PETA_INRIA_h800p:h200u_nh800pp:nh200pat**: 800 perfect and 200 unperfect human samples with 800 perfect+ non-human samples and 200 patches
- **PETA_h250p_nh250patches**: 250 perfect human samples with 250 non-human patches

### 3. Testing Datasets

Three test sets of varying size and quality, created by randomly selecting images from source datasets with a fixed random seed (4402) for reproducibility:

- **perfect_100**: 100 high-quality test samples
  - 50 human samples from PETA
  - 50 non-human samples from INRIA_non_human_perfect_501

- **perfect_200**: 200 high-quality test samples
  - 100 human samples from PETA
  - 100 non-human samples from INRIA_non_human_perfect_501

- **unperfect_200**: 200 test samples including some lower-quality examples
  - 150 human samples from PETA + 50 from INRIA_human_unperfect_301
  - 150 non-human samples from INRIA_non_human_plus_1324 + 50 from patches_non_human

Each test set is divided into human_test and non_human_test folders.
