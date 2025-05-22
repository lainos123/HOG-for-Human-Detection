# this folder will  be used to save the predictions on Testing Images to predictions.xlsx

# for image in Testing Images we need to make a prediction and add it to the predictions.xlsx

import os
import pandas as pd
from test_image import test_image
from pathlib import Path
from utils import extract_hog_params_from_model_name

def save_predictions(image_dir, model_name="svm_hog_classifier_PETA_INRIA_1.joblib", output_file="predictions.xlsx"):
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return

    # Get the base path and construct the model path
    base_path = Path(__file__).resolve().parents[1]  # Go up to Others directory
    model_path = base_path / "Final Model" / model_name
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Extract HOG parameters from model name
    hog_params = extract_hog_params_from_model_name(model_name)

    predictions = []

    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, fname)
            # Get prediction and handle None case, passing HOG parameters
            prediction = test_image(str(model_path), image_path, return_decision_value=False, hog_params=hog_params)
            if prediction is not None:  # Only add valid predictions
                predictions.append({"filename": fname, "prediction": int(prediction)})

    if not predictions:
        print("No valid predictions were made. Please check the model and images.")
        return

    # Create DataFrame with predictions
    df = pd.DataFrame(predictions)
    
    # Add empty rows and model information at the bottom
    empty_rows = pd.DataFrame({"filename": [""] * 2, "prediction": [""] * 2})
    info_rows = pd.DataFrame({
        "filename": ["Model used:", model_name, "", "Images folder:", os.path.abspath(image_dir)],
        "prediction": ["", "", "", "", ""]
    })
    
    # Concatenate all DataFrames
    final_df = pd.concat([df, empty_rows, info_rows], ignore_index=True)

    # Save to Excel
    output_path = os.path.join(os.path.dirname(image_dir), output_file)
    final_df.to_excel(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Model used: {model_name}")
    print(f"Images folder: {os.path.abspath(image_dir)}")