'''
This script creates training datasets by combining images from different raw sources.
For all random operations, the seed is set to 4402 for reproducibility.
'''

import os
import random
import shutil
from pathlib import Path
import argparse

def is_image_file(path):
    """Check if a file is an image based on its extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    return path.suffix.lower() in image_extensions

def create_output_folders(output_path, dataset_names):
    """Create output folders for training data with different combinations"""
    # Dictionary to store all folder paths
    folders = {}
    
    # Create all dataset folders and their human/non-human training subfolders
    for dataset_name in dataset_names:
        dataset_dir = output_path / dataset_name
        
        # Create human and non-human folders
        human_dir = dataset_dir / "human_train"
        non_human_dir = dataset_dir / "non_human_train"
        
        # Add to folders dictionary
        folders[f"{dataset_name}_human"] = human_dir
        folders[f"{dataset_name}_non_human"] = non_human_dir
        
        # Create and clear folders
        for folder in [human_dir, non_human_dir]:
            os.makedirs(folder, exist_ok=True)
            # Clear existing files
            for item in folder.glob('*'):
                if item.is_file():
                    item.unlink()
            print(f"Created and cleared folder: {folder}")
    
    return folders

def get_test_images(test_path):
    """Get set of all image filenames used in test sets to avoid overlap"""
    test_images = set()
    
    # Look for all test folders (perfect and unperfect)
    for test_type in ["perfect_200", "perfect_100", "unperfect_200"]:
        test_dir = test_path / test_type
        
        # Add human and non-human test images
        for category in ["human_test", "non_human_test"]:
            category_path = test_dir / category
            if category_path.exists():
                test_images.update(f.name for f in category_path.glob('*') if f.is_file() and is_image_file(f))
    
    print(f"Found {len(test_images)} images already used in test sets (will be excluded from training)")
    return test_images

def select_train_images(source_folder, count, test_images, seed=4402):
    """
    Select random training images from a source folder, excluding test images
    
    Args:
        source_folder: Source folder path
        count: Number of images to select
        test_images: Set of image names already used in test sets
        seed: Random seed
        
    Returns:
        List of selected image file paths
    """
    random.seed(seed)
    
    # Get all image files from source folder
    all_images = [f for f in source_folder.glob('*') if f.is_file() and is_image_file(f)]
    
    # Filter out test images
    available_images = [f for f in all_images if f.name not in test_images]
    
    print(f"Found {len(all_images)} total images in {source_folder}, {len(available_images)} available after excluding test images")
    
    # Check if we have enough images
    if len(available_images) < count:
        print(f"WARNING: Not enough images in {source_folder}. Need {count}, found {len(available_images)}")
        # Use all available images
        selected_files = available_images
    else:
        # Select random images
        selected_files = random.sample(available_images, count)
    
    print(f"Selected {len(selected_files)} images from {source_folder}")
    return selected_files

def copy_to_destination(images, dest_folder):
    """Copy image files to destination folder"""
    for img in images:
        shutil.copy2(img, dest_folder / img.name)
    
    print(f"Copied {len(images)} images to {dest_folder}")
    return len(images)

def create_training_datasets(base_path, seed=4402):
    """Create training datasets according to the specified requirements"""
    # Set seed for reproducibility
    random.seed(seed)
    
    # Define paths
    datasets_dir = base_path / "Datasets"
    train_dir = datasets_dir / "Train"
    test_dir = datasets_dir / "Test"
    raw_dir = datasets_dir / "Raw"
    
    # Source folders
    peta_humans = raw_dir / "original" / "PETA_human_2023"         # Regular human images (p)
    inria_humans_unperfect = raw_dir / "created" / "INRIA_human_unperfect_301"  # Unperfect human images (u)
    inria_non_humans_plus = raw_dir / "created" / "INRIA_non_human_plus_1324"  # Regular non-human images (pp)
    patches_non_humans = raw_dir / "created" / "patches_non_human"  # Patch non-human images (patches)
    
    # Check if source folders exist
    missing_folders = []
    for folder in [peta_humans, inria_non_humans_plus, patches_non_humans, inria_humans_unperfect]:
        if not folder.exists():
            missing_folders.append(folder)
    
    if missing_folders:
        print("ERROR: The following source folders were not found:")
        for folder in missing_folders:
            print(f"  - {folder}")
        print("\nPlease ensure these folders exist before running this script.")
        return
    
    # Get test images to avoid overlap
    test_images = get_test_images(test_dir)
    
    # Corrected dataset names
    dataset_names = [
        "PETA_INRIA_h250u:h250p_nh500pp",          # 250 unperfect human + 250 regular human + 500 non-human plus
        "PETA_INRIA_h500p_nh500pp",                # 500 regular human + 500 non-human plus
        "PETA_INRIA_h250p_nh250pp",                # 250 regular human + 250 non-human plus
        "PETA_INRIA_h800p:h200u_nh1000pp",         # 800 regular human + 200 unperfect human + 1000 non-human plus
        "PETA_INRIA_h800p:h200u_nh800pp:nh200pat", # 800 regular human + 200 unperfect human + 800 non-human plus + 200 patches
        "PETA_h250p_nh250patches"                  # 250 regular human + 250 patches non-human
    ]
    
    # Create output folders
    folders = create_output_folders(train_dir, dataset_names)
    
    # Process each dataset independently
    
    print("\n--- Creating PETA_INRIA_h250u:h250p_nh500pp dataset ---")
    # 250 unperfect (INRIA) human images
    inria_human_images = select_train_images(inria_humans_unperfect, 250, test_images, seed)
    copy_to_destination(inria_human_images, folders["PETA_INRIA_h250u:h250p_nh500pp_human"])
    
    # 250 regular (PETA) human images
    peta_human_images = select_train_images(peta_humans, 250, test_images, seed + 1)
    copy_to_destination(peta_human_images, folders["PETA_INRIA_h250u:h250p_nh500pp_human"])
    
    # 500 non-human plus images
    inria_non_images = select_train_images(inria_non_humans_plus, 500, test_images, seed + 2)
    copy_to_destination(inria_non_images, folders["PETA_INRIA_h250u:h250p_nh500pp_non_human"])
    
    print("\n--- Creating PETA_INRIA_h500p_nh500pp dataset ---")
    # 500 regular (PETA) human images
    peta_human_images = select_train_images(peta_humans, 500, test_images, seed + 3)
    copy_to_destination(peta_human_images, folders["PETA_INRIA_h500p_nh500pp_human"])
    
    # 500 non-human plus images
    inria_non_images = select_train_images(inria_non_humans_plus, 500, test_images, seed + 4)
    copy_to_destination(inria_non_images, folders["PETA_INRIA_h500p_nh500pp_non_human"])
    
    print("\n--- Creating PETA_INRIA_h250p_nh250pp dataset ---")
    # 250 regular (PETA) human images
    peta_human_images = select_train_images(peta_humans, 250, test_images, seed + 5)
    copy_to_destination(peta_human_images, folders["PETA_INRIA_h250p_nh250pp_human"])
    
    # 250 non-human plus images
    inria_non_images = select_train_images(inria_non_humans_plus, 250, test_images, seed + 6)
    copy_to_destination(inria_non_images, folders["PETA_INRIA_h250p_nh250pp_non_human"])
    print("\n--- Creating PETA_INRIA_h800p:h200u_nh1000pp dataset ---")
    # 800 regular (PETA) human images
    peta_human_images = select_train_images(peta_humans, 800, test_images, seed + 7)
    copy_to_destination(peta_human_images, folders["PETA_INRIA_h800p:h200u_nh1000pp_human"])
    
    # 200 unperfect (INRIA) human images
    inria_human_images = select_train_images(inria_humans_unperfect, 200, test_images, seed + 8)
    copy_to_destination(inria_human_images, folders["PETA_INRIA_h800p:h200u_nh1000pp_human"])
    
    # 1000 non-human plus images
    inria_non_images = select_train_images(inria_non_humans_plus, 1000, test_images, seed + 9)
    copy_to_destination(inria_non_images, folders["PETA_INRIA_h800p:h200u_nh1000pp_non_human"])
    
    print("\n--- Creating PETA_INRIA_h800p:h200u_nh800pp:nh200pat dataset ---")
    # 800 regular (PETA) human images
    peta_human_images = select_train_images(peta_humans, 800, test_images, seed + 10)
    copy_to_destination(peta_human_images, folders["PETA_INRIA_h800p:h200u_nh800pp:nh200pat_human"])
    
    # 200 unperfect (INRIA) human images
    inria_human_images = select_train_images(inria_humans_unperfect, 200, test_images, seed + 11)
    copy_to_destination(inria_human_images, folders["PETA_INRIA_h800p:h200u_nh800pp:nh200pat_human"])
    
    # 800 non-human plus images
    inria_non_images = select_train_images(inria_non_humans_plus, 800, test_images, seed + 12)
    copy_to_destination(inria_non_images, folders["PETA_INRIA_h800p:h200u_nh800pp:nh200pat_non_human"])
    
    # 200 patches non-human images
    patches_images = select_train_images(patches_non_humans, 200, test_images, seed + 13)
    copy_to_destination(patches_images, folders["PETA_INRIA_h800p:h200u_nh800pp:nh200pat_non_human"])
    
    print("\n--- Creating PETA_h250p_nh250patches dataset ---")
    # 250 regular (PETA) human images
    peta_human_images = select_train_images(peta_humans, 250, test_images, seed + 14)
    copy_to_destination(peta_human_images, folders["PETA_h250p_nh250patches_human"])
    
    # 250 patches non-human images
    patches_images = select_train_images(patches_non_humans, 250, test_images, seed + 15)
    copy_to_destination(patches_images, folders["PETA_h250p_nh250patches_non_human"])
    
    # Summary
    print("\n--- Training Dataset Creation Summary ---")
    for dataset_name in dataset_names:
        human_count = len(list(folders[f"{dataset_name}_human"].glob('*')))
        non_human_count = len(list(folders[f"{dataset_name}_non_human"].glob('*')))
        print(f"{dataset_name}:")
        print(f"  - Human training images: {human_count}")
        print(f"  - Non-human training images: {non_human_count}")
        print(f"  - Total: {human_count + non_human_count}")
    
    # Dataset naming convention explanation
    print("\nDataset naming convention:")
    print("- p = human images from PETA_human_2023")
    print("- u = human images from INRIA_human_unperfect_301")
    print("- pp = non-human images from INRIA_non_human_plus_1324")
    print("- patches/pat = non-human images from patches_non_human")

def main():
    parser = argparse.ArgumentParser(description="Create training datasets from raw images")
    parser.add_argument("--seed", type=int, default=4402, 
                        help="Random seed for reproducibility (default: 4402)")
    args = parser.parse_args()
    
    # Base path (root of the project, parent of both "Submission" and "Datasets")
    base_path = Path(__file__).resolve().parents[3]
    
    print(f"Creating training datasets with seed {args.seed}")
    print(f"Base path: {base_path}")
    
    create_training_datasets(base_path, args.seed)
    
    print("\nTraining dataset creation completed!")

if __name__ == "__main__":
    main()
