import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, det_curve
from extract_HOG import compute_hog, test_final_dataset
from pathlib import Path
import argparse

def predict_image(model, image_path, hog_params=None, return_decision_value=True):
    """
    Predict an image directly using a loaded model
    
    Parameters:
    - model: Pre-loaded SVM model
    - image_path: Path to the image
    - hog_params: Dictionary with HOG parameters
    - return_decision_value: Whether to return decision value
    
    Returns:
    - (prediction, decision_value) or just prediction
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features with custom parameters if provided
    if hog_params:
        hog_features = compute_hog(gray, 
                               cell_size=hog_params.get('cell_size', 8),
                               block_size=hog_params.get('block_size', 16),
                               num_bins=hog_params.get('num_bins', 9),
                               angle=hog_params.get('angle', 180))
    else:
        hog_features = compute_hog(gray)

    # Get decision value (distance to hyperplane)
    decision_value = model.decision_function(hog_features.reshape(1, -1))[0]
    
    # Get the actual prediction from the SVM (uses threshold=0)
    prediction = model.predict(hog_features.reshape(1, -1))[0]
    
    # Return based on parameter
    if return_decision_value:
        return prediction, decision_value
    else:
        return prediction

def evaluate_final_model(base_path, test_dataset_name=None, hog_params=None, model_index=None):
    """
    Evaluate the final model on the final test dataset or a specific dataset
    
    Parameters:
    - base_path: Base path of the project
    - test_dataset_name: Optional name of test dataset, if None uses the one from Final Dataset
    - hog_params: Dictionary of HOG parameters to use (if None, uses default from model name)
    - model_index: Optional index of the model to use if multiple exist
    """
    final_model_dir = base_path / "Submission" / "Others" / "Final Model"
    
    if not final_model_dir.exists():
        print(f"Error: Final Model directory not found at {final_model_dir}")
        return None
    
    # Find all model files
    model_files = list(final_model_dir.glob("svm_hog_classifier*.joblib"))
    if not model_files:
        print(f"Error: No model file found in {final_model_dir}")
        return None
    
    # If there are multiple models, prompt user to select one
    if len(model_files) > 1 and model_index is None:
        print(f"\nFound {len(model_files)} models in {final_model_dir}:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file.name}")
        
        while True:
            try:
                selection = input("\nSelect model number to use (or press Enter for first model): ")
                if selection.strip() == "":
                    model_index = 0  # Use first model by default
                    break
                    
                model_idx = int(selection) - 1
                if 0 <= model_idx < len(model_files):
                    model_index = model_idx
                    break
                else:
                    print(f"Please enter a number between 1 and {len(model_files)}")
            except ValueError:
                print("Please enter a valid number")
    elif model_index is None:
        model_index = 0  # If only one model, use it
    
    # Use the selected model
    model_path = model_files[model_index]
    print(f"Using model: {model_path.name}")
    
    # Load the model
    try:
        print(f"Loading model from: {model_path}")
        svm_classifier = joblib.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Extract HOG parameters from model name if not provided
    if hog_params is None:
        hog_params = extract_hog_params(model_path.name)
    
    # Determine the test dataset to use
    if test_dataset_name is None:
        # Try to get the test dataset from Final Dataset
        final_dataset_result = test_final_dataset(base_path)
        if final_dataset_result is None:
            print("Error: No valid test dataset found in Final Dataset directory.")
            print("Using perfect_200 as fallback.")
            test_dataset_name = "perfect_200"
            test_dir = base_path / "Datasets" / "Test"
            human_test_dir = test_dir / test_dataset_name / "human_test"
            non_human_test_dir = test_dir / test_dataset_name / "non_human_test"
        else:
            _, test_dir, _, _ = final_dataset_result
            # In Final Dataset, human_test and non_human_test are directly under Test
            human_test_dir = test_dir / "human_test"
            non_human_test_dir = test_dir / "non_human_test"
            test_dataset_name = "final_test"
    else:
        # Use the specified test dataset from regular Test directory
        test_dir = base_path / "Datasets" / "Test"
        human_test_dir = test_dir / test_dataset_name / "human_test"
        non_human_test_dir = test_dir / test_dataset_name / "non_human_test"

    # Verify directories exist
    if not human_test_dir.exists() or not non_human_test_dir.exists():
        print(f"Error: Test directories not found:")
        print(f"Looking for:\n{human_test_dir}\n{non_human_test_dir}")
        return None

    print(f"\n=== MODEL EVALUATION ON {test_dataset_name} ===")
    print(f"\nModel: {model_path.name}")
    print(f"Model path: {model_path}")
    print(f"Test dataset: {test_dataset_name}")
    print(f"Human test images: {human_test_dir}")
    print(f"Non-human test images: {non_human_test_dir}")
    
    # Print HOG parameters if provided
    if hog_params:
        print("\nUsing HOG parameters:")
        print(f"  Cell size: {hog_params.get('cell_size', 8)}")
        print(f"  Block size: {hog_params.get('block_size', 16)}")
        print(f"  Number of bins: {hog_params.get('num_bins', 9)}")
        print(f"  Angle range: {hog_params.get('angle', 180)}")
    else:
        print("\nUsing default HOG parameters")

    y_true = []
    y_pred = []  # Store SVM predictions
    y_scores = []  # Store decision values
    image_paths = []  # Store paths of processed images
    
    # Track counts for reporting
    human_count = 0
    non_human_count = 0

    # Test positive samples (humans)
    print("\nProcessing human test images...")
    for img_name in os.listdir(human_test_dir):
        img_path = os.path.join(human_test_dir, img_name)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Use direct prediction
            result = predict_image(svm_classifier, img_path, hog_params=hog_params)
            if result is not None:  # Only add valid predictions
                pred, score = result
                y_true.append(1)  # 1 = human
                y_pred.append(pred)
                y_scores.append(score)
                image_paths.append(img_path)  # Store the image path
                human_count += 1
                
                # Show progress for large datasets
                if human_count % 50 == 0:
                    print(f"  Processed {human_count} human images...")
    
    print(f"Processed {human_count} human test images")

    # Test negative samples (non-humans)
    print("\nProcessing non-human test images...")
    for img_name in os.listdir(non_human_test_dir):
        img_path = os.path.join(non_human_test_dir, img_name)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Use direct prediction
            result = predict_image(svm_classifier, img_path, hog_params=hog_params)
            if result is not None:  # Only add valid predictions
                pred, score = result
                y_true.append(0)  # 0 = non-human
                y_pred.append(pred)
                y_scores.append(score)
                image_paths.append(img_path)  # Store the image path
                non_human_count += 1
                
                # Show progress for large datasets
                if non_human_count % 50 == 0:
                    print(f"  Processed {non_human_count} non-human images...")
    
    print(f"Processed {non_human_count} non-human test images")
    print(f"Total test images: {human_count + non_human_count}")

    if not y_true:
        print("No valid predictions were made. Please check the test images and model.")
        return None

    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate accuracy
    accuracy = (tp + tn) / len(y_true)
    
    print(f"\n=== RESULTS FOR {test_dataset_name} ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate ROC curve with sklearn
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create first figure with metrics
    fig_metrics = plt.figure(figsize=(18, 6))
    
    # Create a grid for metrics subplots: 1 row, 3 columns
    gs_metrics = fig_metrics.add_gridspec(1, 3)
    
    # Confusion matrix subplot (first position)
    ax_cm = fig_metrics.add_subplot(gs_metrics[0, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Human", "Human"])
    disp.plot(cmap='Blues', values_format='d', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    
    # ROC curve subplot (second position)
    ax_roc = fig_metrics.add_subplot(gs_metrics[0, 1])
    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f}, Accuracy = {accuracy:.2%})')
    ax_roc.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle='--')
    
    # DET curve subplot (third position)
    ax_det = fig_metrics.add_subplot(gs_metrics[0, 2])
    fpr_det, fnr_det, _ = det_curve(y_true, y_scores)
    fpr_det = np.clip(fpr_det, 1e-6, None)  # Clip to avoid log(0) issues
    
    ax_det.plot(fpr_det, fnr_det, label=f'DET curve (AUC = {roc_auc:.2f}, Accuracy = {accuracy:.2%})')
    ax_det.set_xscale('log')
    ax_det.set_xlabel('False Positive Rate (log scale)')
    ax_det.set_ylabel('False Negative Rate')
    ax_det.set_title('Detection Error Tradeoff (DET) Curve')
    ax_det.set_ylim(0, 0.5)
    ax_det.set_xlim([1e-4, 1e0])
    ax_det.grid(True, linestyle='--')
    ax_det.legend(loc="upper right")
    
    # Add a title to the metrics figure
    fig_metrics.suptitle(f'HOG-SVM Model Performance Metrics', fontsize=16)
    fig_metrics.tight_layout()
    fig_metrics.subplots_adjust(top=0.9)  # Make room for the suptitle
    plt.show()
    
    # Find misclassified images for second figure
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    # Create second figure for misclassified images
    if len(misclassified_indices) > 0:
        # Get misclassified image paths and labels
        misclassified_image_paths = [image_paths[i] for i in misclassified_indices]
        misclassified_true_labels = [y_true[i] for i in misclassified_indices]
        misclassified_pred_labels = [y_pred[i] for i in misclassified_indices]
        
        # Select up to 12 random misclassified images to display
        num_to_show = min(12, len(misclassified_image_paths))
        if num_to_show < len(misclassified_image_paths):
            # Randomly select images
            selected_indices = random.sample(range(len(misclassified_image_paths)), num_to_show)
            selected_images = [misclassified_image_paths[i] for i in selected_indices]
            selected_true = [misclassified_true_labels[i] for i in selected_indices]
            selected_pred = [misclassified_pred_labels[i] for i in selected_indices]
        else:
            selected_images = misclassified_image_paths
            selected_true = misclassified_true_labels
            selected_pred = misclassified_pred_labels
        
        # Determine grid dimensions for misclassified images (roughly square)
        rows = int(np.ceil(np.sqrt(num_to_show)))
        cols = int(np.ceil(num_to_show / rows))
        
        # Create figure for misclassified images
        fig_misclass = plt.figure(figsize=(15, 10))
        
        # Create a subplot for each misclassified image
        for i, (img_path, true_label, pred_label) in enumerate(zip(selected_images, selected_true, selected_pred)):
            if i >= num_to_show:
                break
                
            # Create subplot in grid
            ax = fig_misclass.add_subplot(rows, cols, i + 1)
                
            # Load and display the image
            img = cv2.imread(img_path)
            if img is not None:
                # Convert from BGR to RGB for proper display
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                
                # Add title with true and predicted labels
                true_text = "Human" if true_label == 1 else "Non-Human"
                pred_text = "Human" if pred_label == 1 else "Non-Human"
                ax.set_title(f"True: {true_text}, Pred: {pred_text}", fontsize=10)
                
                # Remove axis ticks
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add a title to the misclassified images figure
        fig_misclass.suptitle(f'Misclassified Examples ({len(misclassified_indices)} total)', fontsize=16)
        fig_misclass.tight_layout()
        fig_misclass.subplots_adjust(top=0.9)  # Make room for the suptitle
        plt.show()
    else:
        print("No misclassified images found!")

def extract_hog_params(model_name):
    """Extract HOG parameters from model filename using a more robust approach"""
    # Default parameters
    hog_params = {
        'cell_size': 8,
        'block_size': 16, 
        'num_bins': 9,
        'block_stride': 1,
        'filter_': 'default',
        'angle': 180
    }
    
    try:
        # Remove the svm_hog_classifier_ prefix if it exists
        name = model_name.replace('svm_hog_classifier_', '')
        
        # Split into parts and look for parameter patterns
        parts = name.split('_')
        for part in parts:
            if part.startswith('c'):
                hog_params['cell_size'] = int(part[1:])
            elif part.startswith('b'):
                hog_params['block_size'] = int(part[1:])
            elif part.startswith('n'):
                hog_params['num_bins'] = int(part[1:])
            elif part.startswith('s'):
                hog_params['block_stride'] = int(part[1:])
            elif part in ['default', 'Sobel', 'Prewitt']:
                hog_params['filter_'] = part
            elif part in ['180', '360']:
                hog_params['angle'] = int(part)
            
        print("Using HOG parameters:")
        print(f"  Cell size: {hog_params['cell_size']}")
        print(f"  Block size: {hog_params['block_size']}")
        print(f"  Number of bins: {hog_params['num_bins']}")
        print(f"  Block stride: {hog_params['block_stride']}")
        print(f"  Filter: {hog_params['filter_']}")
        print(f"  Angle range: {hog_params['angle']}")
        
        return hog_params
    except Exception as e:
        print(f"Error extracting HOG parameters: {e}")
        print("Using default HOG parameters")
        return hog_params  # Return default params on error

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate the final model on test data")
    parser.add_argument("--test-dataset", type=str, help="Name of test dataset to use (if not using Final Dataset)")
    parser.add_argument("--use-final-dataset", action="store_true", help="Use the test dataset from Final Dataset directory")
    parser.add_argument("--model-index", type=int, help="Index of the model to use if multiple exist")
    args = parser.parse_args()
    
    # Set up base path
    base_path = Path(__file__).resolve().parents[3]  # Path to project root
    
    # Determine which test dataset to use
    test_dataset_name = None
    
    # Always try to use the final dataset first
    # Check if the final dataset exists and is valid
    final_dataset_result = test_final_dataset(base_path)
    if final_dataset_result is not None:
        print("Using test dataset from Final Dataset directory")
        # If valid, use the final dataset
        test_dataset_name = None  # Will be determined within evaluate_final_model
    # If not valid or if --test-dataset is explicitly provided, use that
    elif args.test_dataset:
        test_dataset_name = args.test_dataset
        print(f"Using specified test dataset: {test_dataset_name}")
    # Otherwise, default to perfect_200
    else:
        test_dataset_name = "perfect_200"
        print(f"Final dataset not found or invalid. Using default test dataset: {test_dataset_name}")
    
    # Evaluate the final model
    evaluate_final_model(base_path, test_dataset_name, model_index=args.model_index)

if __name__ == "__main__":
    main()