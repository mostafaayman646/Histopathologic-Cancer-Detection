import os
import yaml

def save_model_config(params: dict, model_name: str, output_path: str):
    """
    Saves model parameters and model name to a YAML file.
    """
    # Prepare the data dictionary
    config_data = {
        'model_name': model_name,
        'parameters': params
    }
    
    # Extract the directory path and create it if it doesn't exist
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    # Save the dictionary to the YAML file
    with open(output_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
        
    print(f"Configuration for '{model_name}' saved successfully to '{output_path}'")

# --- Example Usage ---
# if __name__ == "__main__":
#     # Example parameters you might get from your search.best_params_
#     best_params = {
#         'regression__alpha': 0.1,
#         'clf__numeric__poly__degree': 3
#     }
    
#     # Save to the requested path
#     save_model_config(
#         params=best_params, 
#         model_name="RidgeRegression", 
#         output_path="../Config/model_config.yaml"
#     )