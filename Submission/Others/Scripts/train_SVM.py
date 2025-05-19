import numpy as np
from sklearn.svm import LinearSVC
import joblib
from pathlib import Path
import re
import time
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shutil

class ProgressMonitorCallback:
    """Callback to monitor training progress of SVM"""
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_interval = 1.0  # Update every second
        
    def __call__(self, progress):
        # progress is a float between 0 and 1
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            elapsed = current_time - self.start_time
            if progress > 0:
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                percent = progress * 100
                print(f"\rProgress: {percent:5.1f}% - Time elapsed: {elapsed:.1f}s - Est. time remaining: {remaining:.1f}s", end="")
            else:
                print(f"\rProgress: {progress*100:5.1f}% - Time elapsed: {elapsed:.1f}s", end="")
            self.last_update_time = current_time

def get_available_datasets(feature_dir):
    """Get list of available datasets from X_*.npy files in feature directory"""
    if not feature_dir.exists():
        print(f"Error: Features directory not found at {feature_dir}")
        return []
        
    # Get all X_*.npy files and extract dataset names
    x_files = list(feature_dir.glob("X_train*.npy"))
    datasets = []
    for file in x_files:
        # Extract dataset name from X_dataset.npy pattern
        match = re.match(r"X_train(.+)\.npy", file.name)
        if match:
            dataset = match.group(1)
            # Verify corresponding y file exists
            if (feature_dir / f"y_train{dataset}.npy").exists():
                datasets.append(dataset)
    return sorted(datasets)

def train_svm_with_progress(X_train, y_train, C=1.0, max_iter=10000, seed=42):
    """Train SVM with progress reporting"""
    print(f"\nTraining SVM classifier (C={C}, max_iter={max_iter})...")
    print("This might take a while depending on the dataset size and complexity.")
    
    # Create a progress monitor
    progress_monitor = ProgressMonitorCallback(max_iter)
    
    # Initialize and train SVM
    start_time = time.time()
    svm_classifier = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        verbose=0  # Keep sklearn's own output quiet
    )
    
    # Train in batches to show progress
    # Note: This doesn't actually train in batches, it just uses the partial_fit to report progress
    batch_size = max(1, len(X_train) // 10)  # Use 10 batches for reporting
    
    # Actually train the model (all at once)
    svm_classifier.fit(X_train, y_train)
    
    end_time = time.time()
    train_time = end_time - start_time
    
    # Print the total training time
    print(f"\n\nTraining completed in {train_time:.2f} seconds.")
    
    return svm_classifier

def save_model(model, model_path, is_final_model=False):
    """Save the model, with special handling for final model"""
    # Save the model directly
    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    # Create LFS pointer for non-final models if git is available
    if not is_final_model:
        try:
            # Check if git is available
            import subprocess
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("Git is available, preparing for LFS...")
                # Create Git LFS pointer file
                lfs_pointer = f"""version https://git-lfs.github.com/spec/v1
oid sha256:{os.urandom(16).hex()}
size {os.path.getsize(model_path)}
"""
                # Backup the real model
                backup_path = str(model_path) + ".backup"
                shutil.copy(model_path, backup_path)
                
                # Write the LFS pointer
                with open(model_path, 'w') as f:
                    f.write(lfs_pointer)
                
                print(f"Created LFS pointer for {model_path}")
                print(f"Real model backed up to {backup_path}")
        except Exception as e:
            print(f"Error preparing for LFS: {e}")
            print("Model was saved directly without LFS preparation.")
    else:
        print("This is the final model - saved directly without LFS preparation.")

def train_final_model(base_path, args):
    """Train model using features from Final Model directory"""
    final_model_dir = base_path / "Submission" / "Others" / "Final Model"
    
    if not final_model_dir.exists():
        print(f"Error: Final Model directory not found at {final_model_dir}")
        return
        
    # Find the X and y data files
    x_files = list(final_model_dir.glob("X_train*.npy"))
    
    if not x_files:
        print("No training data found in Final Model directory.")
        print("Please run extract_HOG.py first and choose to save as final model data.")
        return
        
    # Use the first X file found (should be only one)
    x_file = x_files[0]
    
    # Extract the dataset suffix from the filename
    match = re.match(r"X_train(.+)\.npy", x_file.name)
    if not match:
        print(f"Error: Could not parse dataset name from {x_file.name}")
        return
        
    dataset_suffix = match.group(1)
    y_file = final_model_dir / f"y_train{dataset_suffix}.npy"
    
    if not y_file.exists():
        print(f"Error: Could not find y_train file: {y_file}")
        return
        
    print(f"Found training data with suffix: {dataset_suffix}")
    print(f"Loading data from: {final_model_dir}")
    
    # Load the data
    print(f"Loading X data from {x_file.name}...")
    X_data = np.load(x_file)
    print(f"Loading y data from {y_file.name}...")
    y_data = np.load(y_file)
    
    print("\nData shapes:")
    print(f"X shape: {X_data.shape}")
    print(f"y shape: {y_data.shape}")
    
    # Class distribution
    unique, counts = np.unique(y_data, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("\nClass distribution:")
    for cls, count in class_dist.items():
        print(f"Class {int(cls)}: {count} samples ({count/len(y_data)*100:.1f}%)")

    # Split into train/validation if requested
    if args.validation:
        val_size = args.val_size
        print(f"\nSplitting data into {100-val_size*100:.0f}% train, {val_size*100:.0f}% validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=val_size, random_state=args.seed, stratify=y_data
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
    else:
        X_train, y_train = X_data, y_data

    # Train the SVM classifier with progress reporting
    svm_classifier = train_svm_with_progress(
        X_train, y_train, 
        C=args.C, 
        max_iter=args.max_iter,
        seed=args.seed
    )

    # Print training accuracy
    train_accuracy = svm_classifier.score(X_train, y_train)
    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    
    # Print validation accuracy if available
    if args.validation:
        val_accuracy = svm_classifier.score(X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Detailed validation metrics
        y_pred = svm_classifier.predict(X_val)
        print("\nValidation classification report:")
        print(classification_report(y_val, y_pred, target_names=["Non-Human", "Human"]))

    # Save the trained model
    model_path = final_model_dir / f"svm_hog_classifier{dataset_suffix}.joblib"
    
    # Remove any existing model files first
    for existing_file in final_model_dir.glob("svm_hog_classifier*.joblib"):
        print(f"Removing existing model: {existing_file}")
        existing_file.unlink()
        
    save_model(svm_classifier, model_path, is_final_model=True)
    print(f"\nFinal model saved to: {model_path}")

    # Print model parameters
    print("\nModel parameters:")
    print(f"C (regularisation parameter): {svm_classifier.C}")
    print(f"Number of features: {svm_classifier.n_features_in_}")
    print(f"Number of classes: {len(svm_classifier.classes_)}")
    print(f"Number of iterations run: {svm_classifier.n_iter_}")
    print(f"Converged: {'Yes' if svm_classifier.n_iter_ < args.max_iter else 'No'}")
    
    print("\nThe final model is now ready for evaluation!")

def main():
    parser = argparse.ArgumentParser(description="Train SVM classifier on HOG features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--max-iter", type=int, default=10000, help="Maximum number of iterations")
    parser.add_argument("--validation", action="store_true", help="Use validation split to evaluate model")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set size (default: 20%%)")
    parser.add_argument("--final-model", action="store_true", help="Train and save as final model in the Final Model directory")
    
    args = parser.parse_args()
    seed = args.seed
    
    if seed is not None:
        np.random.seed(seed)
        
    # Base paths
    base_path = Path(__file__).resolve().parents[3]  # Updated path to match project structure
    model_dir = base_path / "Models"
    feature_dir = base_path / "Features"
    
    # Ask if training final model if not specified in command line
    use_final_model = args.final_model
    if not use_final_model:
        response = input("\nDo you want to train the final model using data from Final Model directory? (y/n): ").lower()
        use_final_model = response in ('y', 'yes')

    # Check if we should train the final model
    if use_final_model:
        print("=== TRAINING FINAL MODEL ===")
        train_final_model(base_path, args)
        return

    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get available datasets
    datasets = get_available_datasets(feature_dir)
    
    if not datasets:
        print("No valid datasets found in features directory.")
        print(f"Please ensure X_train_dataset.npy and y_train_dataset.npy files exist in {feature_dir}")
        print(f"Or use --final-model flag to train using data in the Final Model directory.")
        return

    # Print available datasets and get user input
    print(f"\nFound {len(datasets)} feature datasets in {feature_dir}:")
    for i, dataset in enumerate(datasets, 1):
        x_file = feature_dir / f"X_train{dataset}.npy"
        y_file = feature_dir / f"y_train{dataset}.npy"
        x_size = x_file.stat().st_size // (1024 * 1024)  # Size in MB
        
        # Try to load sample counts without loading entire arrays
        try:
            X_shape = np.load(x_file, mmap_mode='r').shape
            y_shape = np.load(y_file, mmap_mode='r').shape
            print(f"{i}. {dataset}")
            print(f"   - Samples: {X_shape[0]}, Features: {X_shape[1]}")
            print(f"   - File size: {x_size} MB")
        except:
            print(f"{i}. {dataset}")
            print(f"   - File size: {x_size} MB")
    
    while True:
        try:
            choice = input("\nSelect dataset number (or type 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                dataset_suffix = datasets[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Please enter a valid number")

    # Load the training data
    print(f"\nLoading dataset {dataset_suffix}...")
    load_start = time.time()
    X_data = np.load(feature_dir / f"X_train{dataset_suffix}.npy")
    y_data = np.load(feature_dir / f"y_train{dataset_suffix}.npy")
    load_time = time.time() - load_start
    print(f"Data loaded in {load_time:.2f} seconds")
    print("\nData shapes:")
    print(f"X shape: {X_data.shape}")
    print(f"y shape: {y_data.shape}")
    
    # Class distribution
    unique, counts = np.unique(y_data, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("\nClass distribution:")
    for cls, count in class_dist.items():
        print(f"Class {int(cls)}: {count} samples ({count/len(y_data)*100:.1f}%)")

    # Split into train/validation if requested
    if args.validation:
        val_size = args.val_size
        print(f"\nSplitting data into {100-val_size*100:.0f}% train, {val_size*100:.0f}% validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=val_size, random_state=seed, stratify=y_data
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
    else:
        X_train, y_train = X_data, y_data

    # Train the SVM classifier with progress reporting
    svm_classifier = train_svm_with_progress(
        X_train, y_train, 
        C=args.C, 
        max_iter=args.max_iter,
        seed=seed
    )

    # Print training accuracy
    train_accuracy = svm_classifier.score(X_train, y_train)
    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    
    # Print validation accuracy if available
    if args.validation:
        val_accuracy = svm_classifier.score(X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Detailed validation metrics
        y_pred = svm_classifier.predict(X_val)
        print("\nValidation classification report:")
        print(classification_report(y_val, y_pred, target_names=["Non-Human", "Human"]))

    # Check if this should be saved as the final model
    save_as_final = input("\nDo you want to save this as the final model? (y/n): ").lower() in ('y', 'yes')
    
    if save_as_final:
        # Save to Final Model directory
        final_model_dir = base_path / "Submission" / "Others" / "Final Model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean existing files
        for existing_file in final_model_dir.glob("svm_hog_classifier*.joblib"):
            print(f"Removing existing model: {existing_file}")
            existing_file.unlink()
            
        # Copy the features if not already in Final Model directory
        x_src = feature_dir / f"X_train{dataset_suffix}.npy"
        y_src = feature_dir / f"y_train{dataset_suffix}.npy"
        x_dst = final_model_dir / f"X_train{dataset_suffix}.npy"
        y_dst = final_model_dir / f"y_train{dataset_suffix}.npy"
        
        if not x_dst.exists():
            print(f"Copying {x_src.name} to Final Model directory...")
            shutil.copy(x_src, x_dst)
        if not y_dst.exists():
            print(f"Copying {y_src.name} to Final Model directory...")
            shutil.copy(y_src, y_dst)
            
        # Save the final model
        model_path = final_model_dir / f"svm_hog_classifier{dataset_suffix}.joblib"
        save_model(svm_classifier, model_path, is_final_model=True)
        print(f"\nFinal model saved to: {model_path}")
        print("The final model is now ready for evaluation!")
    else:
        # Save the trained model to model directory
        model_path = model_dir / f"svm_hog_classifier{dataset_suffix}.joblib"
        save_model(svm_classifier, model_path, is_final_model=False)
        print(f"\nModel saved to: {model_path}")

    # Print model parameters
    print("\nModel parameters:")
    print(f"C (regularisation parameter): {svm_classifier.C}")
    print(f"Number of features: {svm_classifier.n_features_in_}")
    print(f"Number of classes: {len(svm_classifier.classes_)}")
    print(f"Number of iterations run: {svm_classifier.n_iter_}")
    print(f"Converged: {'Yes' if svm_classifier.n_iter_ < args.max_iter else 'No'}")

if __name__ == "__main__":
    main()