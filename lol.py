
def read_config(file_path):
    # Set default values
    model = "YOLOv8"
    confidence_threshold = 0.5
    
    try:
        # Reading the configuration from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # Process each line and extract values
                if line.startswith("Model:"):
                    model = line.split(":")[1].strip()
                elif line.startswith("Confidence Threshold:"):
                    try:
                        confidence_threshold = float(line.split(":")[1].strip())
                    except ValueError:
                        print("Invalid confidence threshold format, using default value (0.5).")
    except FileNotFoundError:
        print(f"Config file '{file_path}' not found, using default values.")
    except Exception as e:
        print(f"An error occurred while reading the config file: {e}")
    
    return model, confidence_threshold

file_path = 'settingpara.txt'
model_text, confidence_threshold_text = read_config(file_path)

print(model_text)
print