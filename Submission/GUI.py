'''This GUI has the following requirements:
1. Load a directory containing 10 positive and 10 negative images
2. Display each image one by one
3. Show the prediction (human or non-human) for each image
4. Save predictions to a file called predictions.xlsx placed in the primary directory of the submission 
    (first coumn is filename and second column is prediciton)'''

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import sys
import openpyxl

# Add Scripts folder to sys.path
scripts_path = os.path.join(os.path.dirname(__file__), "Others", "Scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import pandas as pd
from test_image import test_image
from pathlib import Path
from save_predictions import save_predictions
from utils import extract_hog_params_from_model_name

class HumanDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection GUI")
        
        # Set initial window size and make it resizable
        self.root.geometry("600x800")
        self.root.resizable(True, True)  # Enable resizing in both directions
        
        # Force window to front
        self.root.deiconify()
        self.root.focus_force()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights to allow resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)  # Make image area expandable
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create widgets
        self.load_button = ttk.Button(self.main_frame, text="Load Directory", command=self.load_directory)
        self.load_button.grid(row=0, column=0, pady=5)
        
        # Create a frame for navigation buttons
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.grid(row=3, column=0, pady=5)
        
        self.prev_button = ttk.Button(self.nav_frame, text="<-- Previous Image", command=self.previous_image)
        self.prev_button.grid(row=0, column=0, padx=5)
        
        self.next_button = ttk.Button(self.nav_frame, text="Next Image -->", command=self.next_image)
        self.next_button.grid(row=0, column=1, padx=5)
        
        # Bind keys
        self.root.bind('<Return>', lambda e: self.load_directory())
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=1, column=0, pady=5)
        
        self.prediction_label = ttk.Label(self.main_frame, text="")
        self.prediction_label.grid(row=4, column=0, pady=5)

        # add an image_name label
        self.image_name_label = ttk.Label(self.main_frame, text="")
        self.image_name_label.grid(row=2, column=0, pady=5)

        # Add a save button
        self.save_button = ttk.Button(self.main_frame, text="Save Predictions", command=self.save_predictions)
        self.save_button.grid(row=5, column=0, pady=5)
        
        # Initialise variables
        self.current_image_index = 0
        self.image_list = []
        self.predictions = {}
        self.current_directory = None  # Add this to store the directory path
        self.current_image = None  # Add this to store the current PhotoImage
        self.model_path = None  # Store the selected model path
        self.root.bind('<Configure>', lambda e: self.display_current_image())
        
        # Initialize by loading the model
        self.load_model()
        
    def load_model(self):
        """Load a model from the Final Model directory"""
        base_path = Path(__file__).resolve().parent
        final_model_dir = base_path / "Others" / "Final Model"
        
        # Find all model files
        model_files = list(final_model_dir.glob("svm_hog_classifier*.joblib"))
        
        if not model_files:
            messagebox.showerror("Error", "No model files found in Final Model directory.")
            return
            
        # If there's only one model, use it
        if len(model_files) == 1:
            self.model_path = model_files[0]
            print(f"Using model: {self.model_path.name}")
            return
            
        # If there are multiple models, prompt user to select one
        model_selection = tk.Toplevel(self.root)
        model_selection.title("Select Model")
        model_selection.geometry("400x300")
        model_selection.transient(self.root)
        model_selection.grab_set()
        
        ttk.Label(model_selection, text="Select a model to use:").pack(pady=10)
        
        # Create a listbox with scrollbar
        frame = ttk.Frame(model_selection)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set)
        for model_file in model_files:
            listbox.insert(tk.END, model_file.name)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=listbox.yview)
        
        # Default select the first item
        listbox.selection_set(0)
        
        def on_select():
            if listbox.curselection():
                index = listbox.curselection()[0]
                self.model_path = model_files[index]
                print(f"Selected model: {self.model_path.name}")
                model_selection.destroy()
            else:
                messagebox.showerror("Error", "Please select a model.")
        
        ttk.Button(model_selection, text="Select", command=on_select).pack(pady=10)
        
        # Wait for the selection window to close
        self.root.wait_window(model_selection)
        
        # If no model was selected, use the first one
        if self.model_path is None and model_files:
            self.model_path = model_files[0]
            print(f"Using default model: {self.model_path.name}")
        
    def load_directory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        directory = filedialog.askdirectory(initialdir=current_dir)
        if directory:
            self.current_directory = directory
            self.image_list = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.current_image_index = 0
            
            # Force window to front after loading images
            self.root.deiconify()
            self.root.focus_force()
            self.display_current_image()
            self.predict()
    
    def display_current_image(self):
        if self.image_list and self.current_image_index < len(self.image_list):
            # Get the current image path
            image_path = os.path.join(self.current_directory, self.image_list[self.current_image_index])
            
            # Open the image
            image = Image.open(image_path)
            
            # Get the current size of the image frame
            frame_width = self.image_label.winfo_width()
            frame_height = self.image_label.winfo_height()
            
            # If the frame hasn't been drawn yet, use default size
            if frame_width <= 1:
                frame_width = 800
                frame_height = 400
            
            # Calculate the resize ratio to maintain aspect ratio
            image_ratio = image.size[0] / image.size[1]
            frame_ratio = frame_width / frame_height
            
            if image_ratio > frame_ratio:
                # Image is wider than frame ratio
                new_width = frame_width
                new_height = int(frame_width / image_ratio)
            else:
                # Image is taller than frame ratio
                new_height = frame_height
                new_width = int(frame_height * image_ratio)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and store reference
            self.current_image = ImageTk.PhotoImage(image)
            
            # Update the label
            self.image_label.config(image=self.current_image)
            
            # Update the prediction label with the filename
            self.image_name_label.config(text=f"Image: {self.image_list[self.current_image_index]}")
    
    def next_image(self):
        if self.image_list:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
            self.display_current_image()
            self.predict()

    def previous_image(self):
        if self.image_list:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
            self.display_current_image()
            self.predict()

    def on_resize(self):
        if self.image_list:
            self.display_current_image()

    def predict(self):
        if not self.image_list or self.current_image_index >= len(self.image_list):
            return
            
        if self.model_path is None:
            messagebox.showerror("Error", "No model selected. Please restart the application.")
            return
            
        image_path = os.path.join(self.current_directory, self.image_list[self.current_image_index])

        # Check image size before predicting
        try:
            with Image.open(image_path) as img:
                if img.size != (64, 128):
                    print(f"ERROR: {self.image_list[self.current_image_index]} is not 64x128, but {img.size}")
        except Exception as e:
            print(f"Failed to read image: {image_path} | Error: {e}")
            return

        # Extract HOG parameters from model name
        hog_params = extract_hog_params_from_model_name(self.model_path.name)
        
        # Call test_image with HOG parameters
        print(f"Testing image {self.image_list[self.current_image_index]} on final SVM classifier using HOG parameters: \n {hog_params}")
        result = test_image(self.model_path, image_path, return_decision_value=False, hog_params=hog_params)

        if result == 1:
            label = "Human"
            colour = "green"
        else:
            label = "Non-Human"
            colour = "red"

        self.prediction_label.config(
            text=f"Prediction: {label}",
            font=("Helvetica", 20, "bold"),
            foreground=colour
        )
        
        # Store the prediction for later saving
        self.predictions[self.image_list[self.current_image_index]] = label
        
    def save_predictions(self):
        """Save predictions to an Excel file"""
        if not self.current_directory or not self.predictions:
            messagebox.showerror("Error", "No predictions to save.")
            return
            
        if self.model_path is None:
            messagebox.showerror("Error", "No model selected.")
            return
            
        try:
            save_predictions(self.current_directory, self.model_path.name)
            messagebox.showinfo("Success", "Predictions saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save predictions: {e}")

def main():
    root = tk.Tk()
    app = HumanDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()