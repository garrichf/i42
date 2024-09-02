import os
import re
import csv

def footage_sort(value):
    """
    Extract numerical value from a folder name for correct numerical sorting based on 'Footage' number.
    """
    numbers = re.findall(r'Footage(\d+)', value)
    return int(numbers[0]) if numbers else 0

def numerical_sort(value):
    """
    Extract numerical value from a string for correct numerical sorting.
    """
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else 0

def find_files(directory, extensions):
    """
    Recursively find files with given extensions in the specified directory,
    and sort them based on the Footage number and then by the numerical order of filenames.
    
    :param directory: The root directory to search in.
    :param extensions: A tuple of file extensions to search for.
    :return: A list of tuples where each tuple contains the file path, the folder name, and the file name.
    """
    files_list = []
    
    # Sort root directories by 'Footage' number
    for root, dirs, files in sorted(os.walk(directory), key=lambda x: footage_sort(os.path.basename(x[0]))):
        # Sort files by frame number
        sorted_files = sorted([f for f in files if f.lower().endswith(extensions)], key=numerical_sort)

        # Extract category from the root directory path
        category = 'NoFall' if 'NoFall' in root else 'Fall' if 'Fall' in root else 'Unknown'
        
        for file_name in sorted_files:
            file_path = os.path.join(root, file_name)
            folder_name = os.path.basename(root)
            # files_list.append((file_path, folder_name, file_name))
            files_list.append((category, file_path, folder_name, file_name))

    return files_list

def save_to_csv(data, output_file):
    """
    Save the list of tuples to a CSV file.
    
    :param data: List of tuples containing file path, folder name, and file name.
    :param output_file: The output CSV file path.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'File Path', 'Folder Name', 'File Name'])
        writer.writerows(data)

# Define the directory to search and the file extensions
directory_to_search = "Dataset"
file_extensions = (".mp4", ".jpeg", ".png",".jpg")

# Find all files with the specified extensions
found_files = find_files(directory_to_search, file_extensions)

# Save the found files to a CSV file
output_csv = "found_files_v2.csv"
save_to_csv(found_files, output_csv)

# Print the found files with their folders
# countfiles = 0
# for category, folder_name, file_name, file_path in found_files:
    # print(f"{countfiles}: Category: {category}, Footage: {folder_name}, File: {file_name}, Path: {file_path}")
    # countfiles += 1
