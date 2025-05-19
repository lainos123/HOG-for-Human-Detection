import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def parse_annotation(annotation_path):
    """Parse XML annotation file and extract person bounding boxes."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    # Get image dimensions
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    # Extract all person bounding boxes
    bboxes = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'person':
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append((xmin, ymin, xmax, ymax))
    
    print(f"Parsing annotation: {annotation_path}")
    print(f"Image size: {width}x{height}")
    print(f"Found {len(bboxes)} person bounding boxes: {bboxes}")
    return width, height, bboxes

def region_overlaps_with_person(region, person_boxes, margin=0):
    """Check if a region overlaps with any person bounding box."""
    x1, y1, x2, y2 = region
    
    for box_x1, box_y1, box_x2, box_y2 in person_boxes:
        # Add margin around person boxes
        box_x1 -= margin
        box_y1 -= margin
        box_x2 += margin
        box_y2 += margin
        
        # Check if there's any overlap
        if not (x2 < box_x1 or x1 > box_x2 or y2 < box_y1 or y1 > box_y2):
            print(f"Region {region} overlaps with person box {(box_x1, box_y1, box_x2, box_y2)}")
            return True
    
    return False

def extract_left_right_negatives(img, width, height, bboxes, margin=5):
    negative_regions = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        person_width = x2 - x1
        person_height = y2 - y1
        print(f"Person {i+1}: bbox=({x1},{y1},{x2},{y2}), size={person_width}x{person_height}")
        
        # Try left region - use same dimensions as person box
        left_x1 = x1 - person_width - margin
        left_x2 = x1 - margin
        left_y1 = y1  # Use same y-coordinates as person
        left_y2 = y2
        print(f"  Trying left region: ({left_x1},{left_y1},{left_x2},{left_y2})")
        if left_x1 >= 0:
            left_region = (left_x1, left_y1, left_x2, left_y2)
            if not region_overlaps_with_person(left_region, bboxes):
                print(f"    Accepted left region: {left_region}")
                negative_regions.append(left_region)
            else:
                print(f"    Rejected left region due to overlap")
        else:
            print(f"    Skipping left region: out of bounds (left_x1={left_x1} < 0)")
            
        # Try right region - use same dimensions as person box
        right_x1 = x2 + margin
        right_x2 = x2 + person_width + margin
        right_y1 = y1  # Use same y-coordinates as person
        right_y2 = y2
        print(f"  Trying right region: ({right_x1},{right_y1},{right_x2},{right_y2})")
        if right_x2 <= width:
            right_region = (right_x1, right_y1, right_x2, right_y2)
            if not region_overlaps_with_person(right_region, bboxes):
                print(f"    Accepted right region: {right_region}")
                negative_regions.append(right_region)
            else:
                print(f"    Rejected right region due to overlap")
        else:
            print(f"    Skipping right region: out of bounds (right_x2={right_x2} > width={width})")
    print(f"Total negative regions found: {len(negative_regions)}")
    return negative_regions

def process_single_image(image_path, annotation_path, output_dir):
    """
    Process a single image and its annotation file to extract negative samples.
    
    Args:
        image_path: Path to the image file
        annotation_path: Path to the XML annotation file
        output_dir: Directory to save negative samples
    
    Returns:
        num_negatives: Number of negative samples extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse annotation
    try:
        width, height, bboxes = parse_annotation(annotation_path)
    except Exception as e:
        print(f"Error parsing annotation {annotation_path}: {e}")
        return 0
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return 0
    
    # Extract negative regions
    negative_regions = extract_left_right_negatives(img, width, height, bboxes)
    
    # Save negative samples
    base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
    
    for i, (x1, y1, x2, y2) in enumerate(negative_regions):
        negative_patch = img[y1:y2, x1:x2]
        
        # Resize to 64x128
        negative_patch = cv2.resize(negative_patch, (64, 128))
        
        if (i % 2) == 0:
            negative_filename = f"{base_name}_L{(i // 2) + 1}.jpg"
        elif (i % 2) == 1:
            negative_filename = f"{base_name}_R{(i // 2) + 1}.jpg"
        else:
            negative_filename = f"{base_name}_{i}.jpg"
        negative_path = os.path.join(str(output_dir), negative_filename)
        
        # Save the negative patch
        cv2.imwrite(negative_path, negative_patch)
    
    print(f"Extracted {len(negative_regions)} negative samples from {os.path.basename(str(image_path))}")
    return len(negative_regions)

def process_single_image_by_name(image_name, dataset_root, split='Train', output_dir='INRIA_Negatives'):
    """
    Process a single image by its name, automatically finding annotation and image paths.
    
    Args:
        image_name: Base name of the image (e.g., 'crop_001009' without extension)
        dataset_root: Root directory of the INRIA dataset
        split: 'Train' or 'Test'
        output_dir: Directory to save negative samples
    
    Returns:
        num_negatives: Number of negative samples extracted
    """
    # Create paths
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    # Try to find the image with various extensions
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_path = dataset_root / split / 'JPEGImages' / f"{image_name}{ext}"
        if potential_path.exists():
            image_path = potential_path
            break
    
    if image_path is None:
        print(f"Error: Could not find image with name {image_name} in {split} set")
        return 0
    
    # Find annotation file
    annotation_path = dataset_root / split / 'Annotations' / f"{image_name}.xml"
    
    if not annotation_path.exists():
        print(f"Error: Could not find annotation file for {image_name}")
        return 0
    
    # Process the image
    return process_single_image(image_path, annotation_path, output_dir)

def process_all_images(dataset_root, output_dir):
    """
    Process all images in the INRIA dataset to extract negative samples.
    
    Args:
        dataset_root: Path to the INRIA dataset root
        output_dir: Directory to save negative samples
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    total_negatives = 0
    
    # After setting up your paths
    print("\nSearching in directories:")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}\n")

    # Process both train and test sets
    for split in ['Train', 'Test']:
        annotations_dir = dataset_root / split / 'Annotations'
        
        annotation_files = list(annotations_dir.glob('*.xml'))
        print(f"Processing {len(annotation_files)} files in {split} set...")
        
        for annotation_file in tqdm(annotation_files):
            # Get image name (without extension)
            image_name = annotation_file.stem
            
            # Process this single image
            num_negatives = process_single_image_by_name(
                image_name, dataset_root, split, output_dir
            )
            total_negatives += num_negatives
    
    print(f"Extracted a total of {total_negatives} negative samples")


if __name__ == "__main__":
    # Instead of using current_dir = Path.cwd()
    script_dir = Path(__file__).parent  # Gets the directory containing this script

    # Then modify the paths
    dataset_root = script_dir.parent / "Dataset" / "Raw" / "INRIAPerson"
    output_dir = script_dir.parent / "Dataset" / "Raw" / "INRIA_non_human"

    # Add prints to verify paths
    print("\nScript location and directories:")
    print(f"Script directory: {script_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}\n")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all images in the dataset
    process_all_images(dataset_root, output_dir)
    print(f"All images processed. Negative samples saved in {output_dir}")