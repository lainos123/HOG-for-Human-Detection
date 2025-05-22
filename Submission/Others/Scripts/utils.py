import re

def extract_hog_params_from_model_name(model_name):
    """Extract HOG parameters from model filename"""
    # Default parameters
    params = {
        'cell_size': 8,
        'block_size': 16,
        'num_bins': 9,
        'block_stride': 1,
        'filter_': 'default',
        'angle': 180
    }
    
    try:
        # Extract parameters using regex patterns
        # Extract cell size (_c4_)
        cell_match = re.search(r'_c(\d+)_', model_name)
        if cell_match:
            params['cell_size'] = int(cell_match.group(1))
        
        # Extract block size (_b32_)
        block_match = re.search(r'_b(\d+)_', model_name)
        if block_match:
            params['block_size'] = int(block_match.group(1))
        
        # Extract number of bins (_n9_)
        bins_match = re.search(r'_n(\d+)_', model_name)
        if bins_match:
            params['num_bins'] = int(bins_match.group(1))
        
        # Extract block stride (_s1_)
        stride_match = re.search(r'_s(\d+)_', model_name)
        if stride_match:
            params['block_stride'] = int(stride_match.group(1))
        
        # Extract angle (180 or 360)
        if '_180' in model_name:
            params['angle'] = 180
        elif '_360' in model_name:
            params['angle'] = 360
        
        print(f"\nExtracted HOG parameters from model {model_name}:")
        
    except Exception as e:
        print(f"Error parsing model parameters: {e}")
        print("Using default parameters")
    
    return params 