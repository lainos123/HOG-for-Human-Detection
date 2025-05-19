import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, det_curve
from test_image import test_image
from pathlib import Path
import re

def get_available_models(model_dir):
    """Get list of available trained models"""
    if not model_dir.exists():
        print(f"Error: Models directory not found at {model_dir}")
        return []
    # Get all svm_hog_classifier*.joblib files
    model_files = list(model_dir.glob("svm_hog_classifier*.joblib"))
    models = []
    for file in model_files:
        # Extract suffix from svm_hog_classifier{suffix}.joblib
        suffix = file.stem.replace("svm_hog_classifier", "")
        models.append(suffix)
    return sorted(models)

def get_available_test_datasets(test_dir):
    """Get list of available test datasets"""
    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        return []
    
    # Get all directories in the test folder
    test_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    # Filter to only include directories that have human_test and non_human_test subdirectories
    valid_datasets = []
    for d in test_dirs:
        if (d / "human_test").exists() and (d / "non_human_test").exists():
            valid_datasets.append(d.name)
    
    return sorted(valid_datasets)

def evaluate_model(model_suffix, test_dataset_name, hog_params=None, return_predictions=False, use_backup_if_available=False):
    """
    Evaluate a trained model on a test dataset
    
    Parameters:
    - model_suffix: Suffix of the model to evaluate
    - test_dataset_name: Name of the test dataset to use
    - hog_params: Dictionary of HOG parameters to use (if None, uses default)
    - return_predictions: If True, return prediction data instead of showing plots
    - use_backup_if_available: If True, try loading .backup file if regular model fails
    
    Returns:
    - If return_predictions=True: Tuple of (y_true, y_scores)
    - If return_predictions=False: None
    """
    base_path = Path(__file__).resolve().parents[3]  # Updated to correctly reach project root
    model_dir = base_path / "Models"
    test_dir = base_path / "Datasets" / "Test"
    
    # Model name
    model_name = f"svm_hog_classifier{model_suffix}.joblib"
    model_path = model_dir / model_name
    backup_path = model_dir / f"{model_name}.backup"
    
    # Check if files exist
    model_exists = model_path.exists()
    backup_exists = backup_path.exists()
    
    if not model_exists and not backup_exists:
        print(f"Error: Neither model file nor backup found at {model_path}")
        return None if return_predictions else None
    
    # Load the model once up front
    import joblib
    svm_model = None
    try:
        print(f"Loading model from {model_path}...")
        svm_model = joblib.load(model_path)
        print(f"Model loaded successfully")
    except Exception as e:
        if use_backup_if_available and backup_exists:
            print(f"Error loading primary model: {e}")
            print(f"Attempting to load backup file: {backup_path}")
            try:
                svm_model = joblib.load(backup_path)
                print(f"Backup model loaded successfully")
            except Exception as e2:
                print(f"Error loading backup file: {e2}")
                return None if return_predictions else None
        else:
            print(f"Error loading model: {e}")
            return None if return_predictions else None
    
    # Set up test dataset paths
    test_dataset_dir = test_dir / test_dataset_name
    
    # Set up human and non-human test directories
    positive_dir = test_dataset_dir / "human_test"
    negative_dir = test_dataset_dir / "non_human_test"

    # Verify directories exist
    if not positive_dir.exists() or not negative_dir.exists():
        print(f"Error: Test directories not found in {test_dataset_dir}")
        print(f"Looking for:\n{positive_dir}\n{negative_dir}")
        return None if return_predictions else None

    print(f"\n=== MODEL EVALUATION ===")
    print(f"\nModel: {model_name}")
    if backup_exists:
        print(f"Backup file available: {backup_path.name}")
    print(f"Test dataset: {test_dataset_name}")
    print(f"Human test images: {positive_dir}")
    print(f"Non-human test images: {negative_dir}")
    
    # Print HOG parameters if provided
    if hog_params:
        print("\nUsing custom HOG parameters:")
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

    # Import compute_hog from extract_HOG to compute HOG features
    from extract_HOG import compute_hog

    # Function to predict directly using our pre-loaded model
    def predict_with_preloaded_model(img_path, return_decision_value=False):
        try:
            # Load and preprocess the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract HOG features with the specified parameters
            if hog_params:
                hog_features = compute_hog(gray, 
                                     cell_size=hog_params.get('cell_size', 8),
                                     block_size=hog_params.get('block_size', 16),
                                     num_bins=hog_params.get('num_bins', 9),
                                     angle=hog_params.get('angle', 180))
            else:
                hog_features = compute_hog(gray)
            
            # Reshape to ensure it's 2D
            hog_features = hog_features.reshape(1, -1)
            
            # Make prediction
            prediction = svm_model.predict(hog_features)[0]
            
            # Get decision value if requested
            decision_value = None
            if return_decision_value:
                decision_value = svm_model.decision_function(hog_features)[0]
                return prediction, decision_value
            
            return prediction
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

    # Test positive samples (humans)
    print("\nProcessing human test images...")
    for img_name in os.listdir(positive_dir):
        img_path = os.path.join(positive_dir, img_name)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Get both the prediction and decision value using our wrapped function
            result = predict_with_preloaded_model(img_path, return_decision_value=True)
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
    for img_name in os.listdir(negative_dir):
        img_path = os.path.join(negative_dir, img_name)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Get both the prediction and decision value using our wrapped function
            result = predict_with_preloaded_model(img_path, return_decision_value=True)
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
        return None if return_predictions else None

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
    
    print(f"\n=== RESULTS ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.4f}")

    # If returning prediction data, skip plotting and return data
    if return_predictions:
        print("Returning prediction data without plotting...")
        return y_true, y_scores

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
    fig_metrics.suptitle(f'HOG-SVM Model Performance Metrics - Model: {model_suffix}', fontsize=16)
    fig_metrics.tight_layout()
    fig_metrics.subplots_adjust(top=0.9)  # Make room for the suptitle
    plt.show()
    
    # Find misclassified images
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
    
    # Print metrics for different thresholds
    print("\nPerformance at different thresholds:")
    print("Threshold\tTPR\t\tFPR\t\tAccuracy")
    print("-" * 50)
    
    # Choose a few key thresholds to report
    key_thresholds = [-2, -1, -0.5, 0, 0.5, 1, 2]
    for threshold in key_thresholds:
        # Make binary predictions using this threshold
        predictions = (y_scores > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (predictions == 1))
        tn = np.sum((y_true == 0) & (predictions == 0))
        fp = np.sum((y_true == 0) & (predictions == 1))
        fn = np.sum((y_true == 1) & (predictions == 0))
        
        # Calculate rates
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        acc = (tp + tn) / len(y_true)
        
        print(f"{threshold:+.2f}\t\t{tpr_val:.4f}\t{fpr_val:.4f}\t{acc:.4f}")
        
        # Mark if this is the same as the SVM's default threshold
        if np.allclose(predictions, y_pred):
            print(f"^ This is equivalent to the SVM's default threshold")

    return None

def get_hog_parameters(model_suffix, dataset_name):
    """
    Extract HOG parameters from model suffix
    
    Parameters:
    - model_suffix: The suffix of the selected model
    - dataset_name: The name of the selected dataset
    """
    # Try to extract HOG parameters from model suffix
    print("\nAttempting to extract HOG parameters from model name...")
    
    try:
        # More robust pattern matching using regular expressions
        
        # Extract cell size (_c16_)
        cell_match = re.search(r'_c(\d+)_', model_suffix)
        if not cell_match:
            raise ValueError("Could not find cell size pattern (_cX_) in model name")
        cell_size = int(cell_match.group(1))
        
        # Extract block size (_b32_)
        block_match = re.search(r'_b(\d+)_', model_suffix)
        if not block_match:
            raise ValueError("Could not find block size pattern (_bX_) in model name")
        block_size = int(block_match.group(1))
        
        # Extract number of bins (_n9_)
        bins_match = re.search(r'_n(\d+)_', model_suffix)
        if not bins_match:
            raise ValueError("Could not find bins pattern (_nX_) in model name")
        num_bins = int(bins_match.group(1))
        
        # Set up HOG parameters
        hog_params = {
            'cell_size': cell_size,
            'block_size': block_size,
            'num_bins': num_bins,
        }
        
        # Try to extract angle (180 or 360)
        if '_180' in model_suffix:
            hog_params['angle'] = 180
        elif '_360' in model_suffix:
            hog_params['angle'] = 360
        else:
            hog_params['angle'] = 180  # Default angle if not specified
            
        print(f"Successfully extracted HOG parameters:")
        print(f"  Cell size: {hog_params['cell_size']}")
        print(f"  Block size: {hog_params['block_size']}")
        print(f"  Number of bins: {hog_params['num_bins']}")
        print(f"  Angle range: {hog_params['angle']}")
        
        return hog_params
            
    except Exception as e:
        print(f"\n⚠️ Error extracting HOG parameters: {e}")
        print("\n⚠️ WARNING: Using incorrect HOG parameters will cause feature dimension mismatches.")
        print("It is critical to use the exact same parameters that were used during model training.")
        print("The model name should contain parameter information like: _c8_b16_n9_s1_180")
        
        # Force manual input since we couldn't extract the parameters
        print("\nYou must specify the HOG parameters that match the model's training:")
        
        hog_params = {}
        
        # Cell size
        while True:
            cell_size = input("Cell size: ").strip()
            try:
                cell_size = int(cell_size)
                if cell_size > 0:
                    break
                else:
                    print("Cell size must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")
        hog_params['cell_size'] = cell_size
        
        # Block size
        while True:
            block_size = input("Block size: ").strip()
            try:
                block_size = int(block_size)
                if block_size > 0:
                    break
                else:
                    print("Block size must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")
        hog_params['block_size'] = block_size
        
        # Number of bins
        while True:
            num_bins = input("Number of bins: ").strip()
            try:
                num_bins = int(num_bins)
                if num_bins > 0:
                    break
                else:
                    print("Number of bins must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")
        hog_params['num_bins'] = num_bins
        
        # Angle range
        while True:
            angle = input("Angle range (180 or 360): ").strip()
            try:
                angle = int(angle)
                if angle in [180, 360]:
                    break
                else:
                    print("Angle must be either 180 or 360.")
            except ValueError:
                print("Please enter a valid integer.")
        hog_params['angle'] = angle
        
        return hog_params

def main():
    base_path = Path(__file__).resolve().parents[3]  # Updated to match project structure
    model_dir = base_path / "Models"
    test_dir = base_path / "Datasets" / "Test"

    # Check if directories exist
    if not model_dir.exists():
        print(f"Error: Models directory not found at {model_dir}")
        return
        
    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        return

    # Get available models
    models = get_available_models(model_dir)
    
    if not models:
        print("No trained models found.")
        print(f"Please ensure svm_hog_classifier*.joblib files exist in {model_dir}")
        return

    # Get available test datasets
    test_datasets = get_available_test_datasets(test_dir)
    
    if not test_datasets:
        print("No valid test datasets found.")
        print(f"Please ensure test datasets with human_test and non_human_test subdirectories exist in {test_dir}")
        return

    # Print available models and get user input
    print("\n=== AVAILABLE MODELS ===")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    # Select model
    model_suffix = None
    while True:
        try:
            choice = input("\nSelect model number (or type 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_suffix = models[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Print available test datasets and get user input
    print("\n=== AVAILABLE TEST DATASETS ===")
    for i, dataset in enumerate(test_datasets, 1):
        human_count = len([f for f in (test_dir / dataset / "human_test").glob("*") if f.is_file()])
        non_human_count = len([f for f in (test_dir / dataset / "non_human_test").glob("*") if f.is_file()])
        total_count = human_count + non_human_count
        print(f"{i}. {dataset} ({total_count} images: {human_count} human, {non_human_count} non-human)")
    
    # Select test dataset
    test_dataset_name = None
    while True:
        try:
            choice = input("\nSelect test dataset number (or type 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(test_datasets):
                test_dataset_name = test_datasets[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(test_datasets)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Get HOG parameters based on model and dataset context
    hog_params = get_hog_parameters(model_suffix, test_dataset_name)

    # Evaluate model with selected test dataset and HOG parameters
    evaluate_model(model_suffix, test_dataset_name, hog_params)

if __name__ == "__main__":
    main()