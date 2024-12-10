import re
import json

def parse_slurm_output(file_path):
    """
    Parse the Slurm .out file and extract information about training/validation progress and top1 accuracy.
    
    Args:
    - file_path (str): Path to the Slurm .out file.
    
    Returns:
    - dict: A list of dictionaries with parsed information.
    """
    data = []

    # Regular expressions to match training and validation lines
    train_pattern = re.compile(r"epoch (\d+):.*(train|val)/top1': ([0-9\.]+).*loss': ([0-9\.]+)")
    model_pattern = re.compile(r"Experiment started: \"(.*)\"\.")
    # epoch_pattern = re.compile(r"Epoch #(\d+)")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    mode = None  # Can be 'train' or 'val'
    model = None
    epoch = None
    for line in lines:
        model_match = model_pattern.search(line)
        if model_match:
            model = "/".join(model_match.group(1).split("/")[2:])
        match = train_pattern.search(line)
        if match:
            # Extract the relevant information from the line
            mode = match.group(2)
            if mode == "train":
                epoch = int(match.group(1))
            top1_accuracy = float(match.group(3))  # Extract the top1 accuracy
            loss = float(match.group(4))  # Extract the top1 accuracy
            
            # Add to the result
            data.append({
                'model': model,
                'mode': mode,
                'epoch': epoch,
                'accuracy': top1_accuracy,
                'loss': loss
            })
    
    return data

def save_to_json(parsed_data, output_path):
    """
    Save parsed data to a JSON file.
    
    Args:
    - parsed_data (list): List of parsed dictionaries.
    - output_path (str): Path to save the JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(parsed_data, json_file, indent=4)

# Example usage:
slurm_file_path = '/home/gridsan/agupta2/6.5940/tiny-training/slurm-mbv2-50kb-miniimagenet.out'
file_identifier = slurm_file_path.split('/')[-1][6:-4]
json_output_path = f'/home/gridsan/agupta2/6.5940/tiny-training/jsons/{file_identifier}.json'

parsed_data = parse_slurm_output(slurm_file_path)
save_to_json(parsed_data, json_output_path)

print(f"Data successfully parsed and saved to {json_output_path}")