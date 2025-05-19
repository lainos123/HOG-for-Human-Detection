'''
This script creates test datasets from raw image folders with specified image counts.
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

def create_output_folders(output_path):
    """Create output folders for test data"""
    # Test directories
    perfect_200_dir = output_path / "Test" / "perfect_200"
    perfect_100_dir = output_path / "Test" / "perfect_100"
    unperfect_dir = output_path / "Test" / "unperfect_200"
    
    folders = {
        # Test folders
        "perfect_200_human": perfect_200_dir / "human_test",
        "perfect_200_non_human": perfect_200_dir / "non_human_test",
        "perfect_100_human": perfect_100_dir / "human_test",
        "perfect_100_non_human": perfect_100_dir / "non_human_test",
        "unperfect_human": unperfect_dir / "human_test",
        "unperfect_non_human": unperfect_dir / "non_human_test",
    }
    
    # Create all folders and clear existing content
    for name, folder in folders.items():
        os.makedirs(folder, exist_ok=True)
        # Clear existing files
        for item in folder.glob('*'):
            if item.is_file():
                item.unlink()
        print(f"Created and cleared folder: {folder}")
    
    return folders

def select_random_images(source_folder, count, seed=4402):
    """
    Select random images from source folder
    
    Args:
        source_folder: Path to source directory
        count: Number of images to select
        seed: Random seed for reproducibility
        
    Returns:
        List of selected image file paths
    """
    random.seed(seed)
    
    # Get all image files from the source folder
    image_files = [f for f in source_folder.glob('*') if f.is_file() and is_image_file(f)]
    
    # Check if we have enough images
    if len(image_files) < count:
        print(f"WARNING: Not enough images in {source_folder}. Needed {count}, found {len(image_files)}")
        return image_files
    
    # Select random images
    selected_files = random.sample(image_files, count)
    print(f"Selected {len(selected_files)} random images from {source_folder}")
    
    return selected_files

def copy_images(images, dest_folder):
    """Copy image files to destination folder"""
    for img in images:
        shutil.copy2(img, dest_folder / img.name)
    
    print(f"Copied {len(images)} images to {dest_folder}")
    return [img.name for img in images]

def create_test_datasets(base_path, seed=4402):
    """Create test datasets according to the specified requirements"""
    # Set seed for reproducibility
    random.seed(seed)
    
    # Define paths
    datasets_dir = base_path / "Datasets"
    test_dir = datasets_dir / "Test"
    raw_dir = datasets_dir / "Raw"
    
    # Source folders
    peta_humans = raw_dir / "original" / "PETA_human_2023"
    inria_non_humans_plus = raw_dir / "created" / "INRIA_non_human_plus_1324"
    patches_non_humans = raw_dir / "created" / "patches_non_human"
    inria_humans_unperfect = raw_dir / "created" / "INRIA_human_unperfect_301"
    
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
    
    # Create output folders
    folders = create_output_folders(datasets_dir)
    
    print("\n--- Step 1: Select images from source folders ---")
    
    # 1. Select 200 random PETA human images
    peta_human_selection = select_random_images(peta_humans, 200, seed)
    
    # 2. Select 200 random INRIA non-human plus images
    inria_non_human_selection = select_random_images(inria_non_humans_plus, 200, seed + 1)
    
    # 3. Select 50 random INRIA human unperfect images
    inria_human_unperfect_selection = select_random_images(inria_humans_unperfect, 50, seed + 2)
    
    # 4. Select 50 random patches non-human images
    patches_non_human_selection = select_random_images(patches_non_humans, 50, seed + 3)
    
    print("\n--- Step 2: Copy selected images to test folders ---")
    
    # Copy all 200 PETA human images to perfect_200
    copy_images(peta_human_selection, folders["perfect_200_human"])
    
    # Copy all 200 INRIA non-human images to perfect_200
    copy_images(inria_non_human_selection, folders["perfect_200_non_human"])
    
    # Copy 100 random PETA human images to perfect_100
    perfect_100_human_selection = random.sample(peta_human_selection, 100)
    copy_images(perfect_100_human_selection, folders["perfect_100_human"])
    
    # Copy 100 random INRIA non-human images to perfect_100
    perfect_100_non_human_selection = random.sample(inria_non_human_selection, 100)
    copy_images(perfect_100_non_human_selection, folders["perfect_100_non_human"])
    
    # Copy 150 random PETA human images to unperfect_200
    unperfect_human_peta_selection = random.sample(peta_human_selection, 150)
    copy_images(unperfect_human_peta_selection, folders["unperfect_human"])
    
    # Copy all 50 INRIA human unperfect images to unperfect_200
    copy_images(inria_human_unperfect_selection, folders["unperfect_human"])
    
    # Copy 150 random INRIA non-human images to unperfect_200
    unperfect_non_human_inria_selection = random.sample(inria_non_human_selection, 150)
    copy_images(unperfect_non_human_inria_selection, folders["unperfect_non_human"])
    
    # Copy all 50 patches non-human images to unperfect_200
    copy_images(patches_non_human_selection, folders["unperfect_non_human"])
    
    # Verify overlap between perfect_100 and perfect_200
    perfect_100_human_names = set(f.name for f in folders["perfect_100_human"].glob('*') if f.is_file())
    perfect_200_human_names = set(f.name for f in folders["perfect_200_human"].glob('*') if f.is_file())
    human_overlap = perfect_100_human_names.intersection(perfect_200_human_names)
    
    perfect_100_non_human_names = set(f.name for f in folders["perfect_100_non_human"].glob('*') if f.is_file())
    perfect_200_non_human_names = set(f.name for f in folders["perfect_200_non_human"].glob('*') if f.is_file())
    non_human_overlap = perfect_100_non_human_names.intersection(perfect_200_non_human_names)
    
    # Summary
    print("\n--- Test Dataset Creation Summary ---")
    print(f"perfect_100/human_test: {len(list(folders['perfect_100_human'].glob('*')))} images")
    print(f"perfect_100/non_human_test: {len(list(folders['perfect_100_non_human'].glob('*')))} images")
    print(f"perfect_200/human_test: {len(list(folders['perfect_200_human'].glob('*')))} images")
    print(f"perfect_200/non_human_test: {len(list(folders['perfect_200_non_human'].glob('*')))} images")
    print(f"unperfect_200/human_test: {len(list(folders['unperfect_human'].glob('*')))} images (150 PETA + 50 INRIA unperfect)")
    print(f"unperfect_200/non_human_test: {len(list(folders['unperfect_non_human'].glob('*')))} images (150 INRIA non-human + 50 patches)")
    
    print(f"\nOverlap between perfect_100 and perfect_200 (human): {len(human_overlap)} images")
    print(f"Overlap between perfect_100 and perfect_200 (non-human): {len(non_human_overlap)} images")
    
    # Calculate total unique test images
    all_test_images = set()
    for folder in folders.values():
        all_test_images.update(f.name for f in folder.glob('*') if f.is_file())
    
    print(f"\nTotal unique test images: {len(all_test_images)}")

def main():
    parser = argparse.ArgumentParser(description="Create test datasets from raw images")
    parser.add_argument("--seed", type=int, default=4402, 
                        help="Random seed for reproducibility (default: 4402)")
    args = parser.parse_args()
    
    # Base path (root of the project, parent of both "Submission" and "Datasets")
    base_path = Path(__file__).resolve().parents[3]
    
    print(f"Creating test datasets with seed {args.seed}")
    print(f"Base path: {base_path}")
    
    create_test_datasets(base_path, args.seed)
    
    print("\nTest dataset creation completed!")

if __name__ == "__main__":
    main()
