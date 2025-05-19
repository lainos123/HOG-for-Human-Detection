# Model Naming Convention

The model filenames follow a standardized format that captures both the dataset used for training and the HOG parameters.

## Format

`svm_hog_classifier_[DATASET]_[HOG_PARAMETERS].joblib`

Where:

### Dataset Component

`[DATASET]` follows the format: `SOURCE_h[COUNT][TYPE]_nh[COUNT][TYPE]` where:
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

Examples:
- `PETA_INRIA_h250p_nh250pp`: 250 perfect human samples with 250 perfect+ non-human samples
- `PETA_INRIA_h800p:h200u_nh800pp:nh200pat`: 800 perfect and 200 unperfect human samples with 800 perfect+ non-human samples and 200 patches

### HOG Parameters Component

`[HOG_PARAMETERS]` follows the format: `c[CELL_SIZE]_b[BLOCK_SIZE]_n[NUM_BINS]_s[BLOCK_STRIDE]_[FILTER]_[ANGLE]` where:
- `c`: Cell size in pixels
- `b`: Block size in cells
- `n`: Number of orientation bins
- `s`: Block stride in cells
- `[FILTER]`: Filter type (default or custom)
- `[ANGLE]`: Maximum angle in degrees (typically 180 or 360)

Examples:
- `c8_b16_n9_s1_default_180`: 8×8 cells, 16×16 blocks, 9 orientation bins, stride of 1, default filter, 180° range
- `c4_b32_n9_s1_180`: 4×4 cells, 32×32 blocks, 9 orientation bins, stride of 1, custom filter, 180° range

## Complete Examples

- `svm_hog_classifier_PETA_INRIA_h250p_nh250pp_c8_b16_n9_s1_default_180.joblib`
  - Dataset: PETA and INRIA with 250 perfect human samples and 250 perfect+ non-human samples
  - HOG: 8×8 cells, 16×16 blocks, 9 orientation bins, stride of 1, default filter, 180° orientation range

- `svm_hog_classifier_PETA_INRIA_h800p:h200u_nh1000pp_c8_b16_n9_s1_default_180.joblib`
  - Dataset: PETA and INRIA with 800 perfect and 200 unperfect human samples and 1000 perfect+ non-human samples
  - HOG: 8×8 cells, 16×16 blocks, 9 orientation bins, stride of 1, default filter, 180° orientation range