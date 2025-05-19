from pathlib import Path
import joblib
import cv2
import os
import numpy as np
from extract_HOG import compute_hog

def test_image(model_name, image_path, return_decision_value=True, hog_params=None, custom_model_loader=None):
    """
    Test an image with a trained model
    
    Parameters:
    - model_name: Name or path of the model file
    - image_path: Path to the image
    - return_decision_value: If True, returns (prediction, decision_value), otherwise just prediction
    - hog_params: Dictionary with HOG parameters (cell_size, block_size, num_bins, angle)
    - custom_model_loader: Optional function to load the model from a custom path
    """
    # Load the trained SVM model
    if custom_model_loader:
        # Use the custom loader function
        svm_classifier = custom_model_loader(model_name)
        if svm_classifier is None:
            return None
    else:
        # Use the default loading approach
        model_path = Path(__file__).resolve().parents[1] / "Models" / model_name
        try:
            svm_classifier = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            return None

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
                                 block_size=hog_params.get('block_size', 2),
                                 num_bins=hog_params.get('num_bins', 9),
                                 angle=hog_params.get('angle', 180))
    else:
        hog_features = compute_hog(gray)

    # Get decision value (distance to hyperplane)
    decision_value = svm_classifier.decision_function(hog_features.reshape(1, -1))[0]
    
    # Get the actual prediction from the SVM (uses threshold=0)
    prediction = svm_classifier.predict(hog_features.reshape(1, -1))[0]
    
    # Print the prediction result
    if prediction == 1:
        print(f"The image {os.path.basename(image_path)} is classified as Human.")
    else:
        print(f"The image {os.path.basename(image_path)} is classified as Non-Human.")
    
    # Return based on parameter
    if return_decision_value:
        return prediction, decision_value
    else:
        return prediction

# if __name__ == "__main__":
#     model = "svm_hog_classifier.joblib"  # Specify the model file name
#     image_path = "/Users/lainemulvay/Desktop/Projects/UNI/cits4402/Research Proj/HOG-for-Human-Detection/Submission/Others/Dataset/Final/non_human_test/crop001040_L1.jpg"  # Specify the image path
#     prediction = test_image(model, image_path)
#     print(f"Prediction: {prediction}")