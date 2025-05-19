import os
import random
import numpy as np
import cv2
from pathlib import Path
import argparse
import shutil

# ----------------- HOG Computation Functions -----------------

def compute_grad(np_image, filter_="default"):
    image = np.float32(np_image)
    if filter_ == "Sobel":
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    elif filter_ == "Prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    else:
        kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)
    mag, ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    return mag, ang

def compute_histogram(mag_cell, ang_cell, num_bins=9, angle=180):
    bin_size = angle // num_bins
    histogram = np.zeros(num_bins)
    for i in range(mag_cell.shape[0]):
        for j in range(mag_cell.shape[1]):
            mag = mag_cell[i, j]
            ang = ang_cell[i, j] % angle if angle == 180 else ang_cell[i, j]
            bin_ = int(ang // bin_size) % num_bins
            histogram[bin_] += mag
    return histogram

def compute_hog(image, cell_size=8, block_size=16, num_bins=9, block_stride=1, filter_="default", angle=180):
    height, width = image.shape
    mag, ang = compute_grad(image, filter_)
    number_of_Xcells = width // cell_size
    number_of_Ycells = height // cell_size
    cell_histograms = np.zeros((number_of_Ycells, number_of_Xcells, num_bins))

    for y in range(number_of_Ycells):
        for x in range(number_of_Xcells):
            y1, y2 = y * cell_size, y * cell_size + cell_size
            x1, x2 = x * cell_size, x * cell_size + cell_size
            mag_cell = mag[y1:y2, x1:x2]
            ang_cell = ang[y1:y2, x1:x2]
            cell_histograms[y, x, :] = compute_histogram(mag_cell, ang_cell, num_bins, angle)

    block_number = block_size // cell_size
    block_hist = []
    for y in range(0, number_of_Ycells - block_number + 1, block_stride):
        for x in range(0, number_of_Xcells - block_number + 1, block_stride):
            block = cell_histograms[y:y+block_number, x:x+block_number, :].flatten()
            eps = 1e-5
            block_norm_1 = block / (np.linalg.norm(block) + eps)
            block_norm_1 = np.clip(block_norm_1, 0, 0.2)
            block_norm_2 = block_norm_1 / (np.linalg.norm(block_norm_1) + eps)
            block_hist.append(block_norm_2)

    return np.concatenate(block_hist)

# ----------------- Feature Extraction Workflow -----------------

def is_image_file(path):
    """Check if a file is an image based on its extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    return path.suffix.lower() in image_extensions

def load_and_extract_hog_features(image_paths, cell_size=8, block_size=16, num_bins=9, block_stride=1, filter_="default", angle=180):
    # Filter out non-image files
    image_paths = [path for path in image_paths if is_image_file(Path(path))]
    
    features = []
    print(f"Found {len(image_paths)} valid image files to process.")
    
    if len(image_paths) == 0:
        print("No valid image files found. Check the input directory:")
        print(image_paths)
        return np.array(features)

    for i, img_path in enumerate(image_paths):
        if i > 0 and i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} images")
            
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = compute_hog(gray, cell_size, block_size, num_bins, block_stride, filter_, angle)
        if hog_features.size == 0:
            print(f"No features extracted from: {img_path} (gray shape: {gray.shape})")
        features.append(hog_features)

    print(f"\nFinished processing {len(features)} images.\n")
    return np.array(features)

def get_available_training_datasets(train_path):
    """Get list of available training datasets"""
    if not train_path.exists():
        print(f"Error: Train path not found at {train_path}")
        return []
        
    # Look for directories containing both human_train and non_human_train subdirectories
    datasets = []
    for item in train_path.iterdir():
        if item.is_dir():
            human_dir = item / "human_train"
            non_human_dir = item / "non_human_train"
            # Check if directories exist and contain at least one valid image file
            if (human_dir.exists() and non_human_dir.exists() and
                any(is_image_file(f) for f in human_dir.iterdir() if f.is_file()) and
                any(is_image_file(f) for f in non_human_dir.iterdir() if f.is_file())):
                datasets.append(item.name)
    return sorted(datasets)

def get_available_test_datasets(test_path):
    """Get list of available test datasets"""
    if not test_path.exists():
        print(f"Error: Test path not found at {test_path}")
        return []
        
    # Look for directories containing both human_test and non_human_test subdirectories
    datasets = []
    for item in test_path.iterdir():
        if item.is_dir():
            human_dir = item / "human_test"
            non_human_dir = item / "non_human_test"
            # Check if directories exist and contain at least one valid image file
            if (human_dir.exists() and non_human_dir.exists() and
                any(is_image_file(f) for f in human_dir.iterdir() if f.is_file()) and
                any(is_image_file(f) for f in non_human_dir.iterdir() if f.is_file())):
                datasets.append(item.name)
    return sorted(datasets)

def get_folder_image_count(folder_path):
    """Count the number of valid image files in a folder"""
    if not folder_path.exists():
        return 0
    return sum(1 for f in folder_path.glob("*") if is_image_file(f))

def get_hog_parameters():
    """Get HOG parameters from user"""
    default_cell_size = 8
    default_block_size = 16
    default_num_bins = 9
    default_block_stride = 1
    default_filter = "default"
    default_angle = 180
    
    while True:
        use_defaults = input("Use default HOG parameters? (y/n): ").lower()
        if use_defaults in ('y', 'yes'):
            return default_cell_size, default_block_size, default_num_bins, default_block_stride, default_filter, default_angle
        elif use_defaults in ('n', 'no'):
            try:
                cell_size = int(input(f"Cell size (default: {default_cell_size}): ") or default_cell_size)
                block_size = int(input(f"Block size (default: {default_block_size}): ") or default_block_size)
                num_bins = int(input(f"Number of bins (default: {default_num_bins}): ") or default_num_bins)
                block_stride = int(input(f"Block stride (default: {default_block_stride}): ") or default_block_stride)
                
                print("Filter options: default, Sobel, Prewitt")
                filter_ = input(f"Filter (default: {default_filter}): ") or default_filter
                if filter_ not in ["default", "Sobel", "Prewitt"]:
                    print(f"Invalid filter. Using default: {default_filter}")
                    filter_ = default_filter
                
                angle = int(input(f"Angle (180 or 360, default: {default_angle}): ") or default_angle)
                if angle not in [180, 360]:
                    print(f"Invalid angle. Using default: {default_angle}")
                    angle = default_angle
                
                return cell_size, block_size, num_bins, block_stride, filter_, angle
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        else:
            print("Please enter 'y' or 'n'.")

def test_final_dataset(base_path):
    """
    Function to check and verify the final dataset structure.
    This can be called from evaluate_final_model or extract_HOG.
    
    Returns:
    - A tuple of (train_dir, test_dir, train_dataset_name, test_dataset_name) or None if invalid
    """
    # Define paths to final dataset
    final_dataset_dir = base_path / "Submission" / "Others" / "Final Dataset"
    
    if not final_dataset_dir.exists():
        print(f"Error: Final Dataset directory not found at {final_dataset_dir}")
        print("Creating directory structure...")
        final_dataset_dir.mkdir(parents=True, exist_ok=True)
        (final_dataset_dir / "Train").mkdir(exist_ok=True)
        (final_dataset_dir / "Test").mkdir(exist_ok=True)
        return None
    
    # Check for Train and Test subdirectories
    train_dir = final_dataset_dir / "Train"
    test_dir = final_dataset_dir / "Test"
    
    if not train_dir.exists():
        print(f"Error: Train directory not found at {train_dir}")
        train_dir.mkdir(exist_ok=True)
        return None
        
    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        test_dir.mkdir(exist_ok=True)
        return None
        
    # Check for direct human_train and non_human_train directories
    human_train_dir = train_dir / "human_train"
    non_human_train_dir = train_dir / "non_human_train"
    
    human_test_dir = test_dir / "human_test"
    non_human_test_dir = test_dir / "non_human_test"
    
    # Check if directories exist and contain images
    train_valid = (human_train_dir.exists() and non_human_train_dir.exists() and
                  any(is_image_file(f) for f in human_train_dir.iterdir() if f.is_file()) and
                  any(is_image_file(f) for f in non_human_train_dir.iterdir() if f.is_file()))
                  
    test_valid = (human_test_dir.exists() and non_human_test_dir.exists() and
                 any(is_image_file(f) for f in human_test_dir.iterdir() if f.is_file()) and
                 any(is_image_file(f) for f in non_human_test_dir.iterdir() if f.is_file()))
    
    if not train_valid:
        print(f"Error: Train directory structure is incomplete at {train_dir}")
        print("Make sure there are 'human_train' and 'non_human_train' subdirectories with images.")
        return None
        
    if not test_valid:
        print(f"Error: Test directory structure is incomplete at {test_dir}")
        print("Make sure there are 'human_test' and 'non_human_test' subdirectories with images.")
        return None
    
    # Create artificial dataset names for train and test
    train_dataset_name = "final_train"
    test_dataset_name = "final_test"
    
    # Print dataset information
    human_train_count = get_folder_image_count(human_train_dir)
    non_human_train_count = get_folder_image_count(non_human_train_dir)
    
    human_test_count = get_folder_image_count(human_test_dir)
    non_human_test_count = get_folder_image_count(non_human_test_dir)
    
    print(f"\n=== FINAL DATASET SUMMARY ===")
    print(f"Train dataset:")
    print(f"  - Human samples: {human_train_count}")
    print(f"  - Non-human samples: {non_human_train_count}")
    print(f"Test dataset:")
    print(f"  - Human samples: {human_test_count}")
    print(f"  - Non-human samples: {non_human_test_count}")
    
    return train_dir, test_dir, train_dataset_name, test_dataset_name

def process_final_dataset(base_path, cell_size, block_size, num_bins, block_stride, filter_, angle):
    """Process the final dataset for feature extraction"""
    # Check the final dataset structure
    result = test_final_dataset(base_path)
    if result is None:
        print("Error with final dataset. Please setup the final dataset structure first.")
        return
        
    train_dir, test_dir, train_dataset_name, test_dataset_name = result
    
    # Set up paths directly to human_train and non_human_train
    human_train_dir = train_dir / "human_train"
    non_human_train_dir = train_dir / "non_human_train"
    
    # Get image paths
    human_train_images = [f for f in human_train_dir.glob("*") if is_image_file(f)]
    non_human_train_images = [f for f in non_human_train_dir.glob("*") if is_image_file(f)]
    
    # Extract HOG features
    print("\nExtracting HOG features from human training images...")
    human_features = load_and_extract_hog_features(
        human_train_images, cell_size, block_size, num_bins, block_stride, filter_, angle
    )

    print("Extracting HOG features from non-human training images...")
    non_human_features = load_and_extract_hog_features(
        non_human_train_images, cell_size, block_size, num_bins, block_stride, filter_, angle
    )

    # Create labels
    human_labels = np.ones(len(human_features))
    non_human_labels = np.zeros(len(non_human_features))

    # Combine features and labels
    X = np.vstack([human_features, non_human_features])
    y = np.concatenate([human_labels, non_human_labels])
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Create suffix with dataset name and HOG parameters
    hog_suffix = f"c{cell_size}_b{block_size}_n{num_bins}_s{block_stride}_{filter_}_{angle}"
    suffix = f"{train_dataset_name}_{hog_suffix}"
    
    # Save to Final Model directory
    final_model_dir = base_path / "Submission" / "Others" / "Final Model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing files in the final model directory
    for existing_file in final_model_dir.glob("*.npy"):
        print(f"Removing existing file: {existing_file}")
        existing_file.unlink()
    
    # Save the features
    np.save(final_model_dir / f"X_train_{suffix}.npy", X)
    np.save(final_model_dir / f"y_train_{suffix}.npy", y)
    
    print(f"\nSaved to {final_model_dir}:")
    print(f"- X_train_{suffix}.npy")
    print(f"- y_train_{suffix}.npy")
    
    print("\nFeatures extracted from Final Dataset and saved to Final Model directory.")
    print("Use train_SVM.py with the --final-model flag to train the final model.")
    
    return suffix

def main(seed=42):
    parser = argparse.ArgumentParser(description="Extract HOG features from training dataset images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--final-dataset", action="store_true", help="Use the final dataset from Submission/Others/Final Dataset")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(seed)

    # Set up paths
    base_path = Path(__file__).resolve().parents[3]
    
    # Ask if using final dataset if not specified in command line
    use_final_dataset = args.final_dataset
    if not use_final_dataset:
        response = input("\nDo you want to use the final dataset from Submission/Others/Final Dataset? (y/n): ").lower()
        use_final_dataset = response in ('y', 'yes')
    
    if use_final_dataset:
        print("\n=== USING FINAL DATASET ===")
        # Get HOG parameters first
        cell_size, block_size, num_bins, block_stride, filter_, angle = get_hog_parameters()
        
        # Process the final dataset
        process_final_dataset(base_path, cell_size, block_size, num_bins, block_stride, filter_, angle)
        return
    
    # Standard dataset processing
    datasets_dir = base_path / "Datasets"
    train_dir = datasets_dir / "Train"
    test_dir = datasets_dir / "Test"
    feature_dir = base_path / "Features"
    final_model_dir = base_path / "Submission" / "Others" / "Final Model"
    
    # Create directories if they don't exist
    feature_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Print dataset counts
    print("\n=== DATASET SUMMARY ===")
    
    # Get available training datasets
    train_datasets = get_available_training_datasets(train_dir)
    if not train_datasets:
        print("\nNo valid training datasets found.")
        print(f"Please ensure {train_dir} has subdirectories with 'human_train' and 'non_human_train' folders containing images.")
        return
        
    print(f"\nFound {len(train_datasets)} training datasets in {train_dir}:")
    for name in train_datasets:
        dataset_path = train_dir / name
        human_count = get_folder_image_count(dataset_path / "human_train")
        non_human_count = get_folder_image_count(dataset_path / "non_human_train")
        print(f"- {name}: {human_count + non_human_count} images ({human_count} human, {non_human_count} non-human)")
        
    # Get available test datasets
    test_datasets = get_available_test_datasets(test_dir)
    if test_datasets:
        print(f"\nFound {len(test_datasets)} test datasets in {test_dir}:")
        for name in test_datasets:
            dataset_path = test_dir / name
            human_count = get_folder_image_count(dataset_path / "human_test")
            non_human_count = get_folder_image_count(dataset_path / "non_human_test")
            print(f"- {name}: {human_count + non_human_count} images ({human_count} human, {non_human_count} non-human)")
    else:
        print("\nNo valid test datasets found.")

    # Print available datasets and get user input for training
    print("\n=== FEATURE EXTRACTION ===")
    print("\nAvailable training datasets:")
    for i, dataset in enumerate(train_datasets, 1):
        dataset_path = train_dir / dataset
        human_train_count = get_folder_image_count(dataset_path / "human_train")
        non_human_train_count = get_folder_image_count(dataset_path / "non_human_train")
        total_count = human_train_count + non_human_train_count
        
        print(f"{i}. {dataset} ({total_count} total images)")
        print(f"   - human_train: {human_train_count} images")
        print(f"   - non_human_train: {non_human_train_count} images")
    
    while True:
        try:
            choice = input("\nSelect training dataset number (or type 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(train_datasets):
                dataset_name = train_datasets[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(train_datasets)}")
        except ValueError:
            print("Please enter a valid number")

    # Get HOG parameters
    cell_size, block_size, num_bins, block_stride, filter_, angle = get_hog_parameters()
    
    # Set up paths for the selected dataset
    dataset_path = train_dir / dataset_name
    human_path = dataset_path / "human_train"
    non_human_path = dataset_path / "non_human_train"
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Human images path: {human_path}")
    print(f"Non-human images path: {non_human_path}")
    print(f"HOG Parameters: Cell Size={cell_size}, Block Size={block_size}, Bins={num_bins}, Stride={block_stride}, Filter={filter_}, Angle={angle}")

    # Get only image files
    human_images = [f for f in human_path.glob("*") if is_image_file(f)]
    non_human_images = [f for f in non_human_path.glob("*") if is_image_file(f)]

    # Print detailed image counts
    print(f"\nHuman training images found: {len(human_images)}")
    print(f"Non-human training images found: {len(non_human_images)}")
    print(f"Total training images: {len(human_images) + len(non_human_images)}\n")

    print("Extracting HOG features from human images...")
    human_features = load_and_extract_hog_features(human_images, cell_size, block_size, num_bins, block_stride, filter_, angle)

    print("Extracting HOG features from non-human images...")
    non_human_features = load_and_extract_hog_features(non_human_images, cell_size, block_size, num_bins, block_stride, filter_, angle)

    human_labels = np.ones(len(human_features))
    non_human_labels = np.zeros(len(non_human_features))

    X = np.vstack([human_features, non_human_features])
    y = np.concatenate([human_labels, non_human_labels])
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Create suffix with dataset name and HOG parameters
    hog_suffix = f"c{cell_size}_b{block_size}_n{num_bins}_s{block_stride}_{filter_}_{angle}"
    suffix = f"{dataset_name}_{hog_suffix}"
    
    # Check if this should be saved as the final model data
    is_final_model = input("\nSave this as the final model data? (y/n): ").lower() in ('y', 'yes')
    
    # Save features
    if is_final_model:
        # Save to Final Model directory
        output_dir = final_model_dir
        # Clear any existing files in the final model directory
        for existing_file in output_dir.glob("*.npy"):
            print(f"Removing existing file: {existing_file}")
            existing_file.unlink()
    else:
        # Save to Features directory
        output_dir = feature_dir
    
    # Save the features
    np.save(output_dir / f"X_train_{suffix}.npy", X)
    np.save(output_dir / f"y_train_{suffix}.npy", y)
    
    print(f"\nSaved to {output_dir}:")
    print(f"- X_train_{suffix}.npy")
    print(f"- y_train_{suffix}.npy")
    
    if is_final_model:
        print("\nThis is now set as the FINAL MODEL data for training and evaluation!")
        print("Use train_SVM.py with the --final-model flag to train the final model.")
    else:
        print("\nUse train_SVM.py to train a model with these features.")
    
    print("\nFeature extraction completed successfully!")

if __name__ == "__main__":
    main()