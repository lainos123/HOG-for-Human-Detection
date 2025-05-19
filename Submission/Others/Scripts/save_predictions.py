# this folder will  be used to save the predictions on Testing Images to predictions.xlsx

# for image in Testing Images we need to make a prediction and add it to the predictions.xlsx

import os
import pandas as pd
from test_image import test_image

def save_predictions(image_dir, model_name="svm_hog_classifier_PETA_INRIA_1.joblib", output_file="predictions.xlsx"):
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return

    predictions = []

    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, fname)
            prediction = test_image(model_name, image_path)
            predictions.append({"filename": fname, "prediction": prediction})

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