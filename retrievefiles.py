import os
import re
import csv

'''
This code is currently made to work with Leons first version of the dataset, have not been altered to handle the new
and proper file naming conventions yet.
'''

def extract_last_three_digits(filename):
    """
    Extract the last three digits from a filename, including leading zeros.
    
    :param filename: The filename from which to extract the digits.
    :return: The extracted digits as a string or None if not found.
    """
    match = re.search(r'(\d{3})(?=\.\w+$)', filename)
    return match.group(0) if match else None

def extract_number_from_string(s):
    """
    Extract the first numerical part from a string.
    
    :param s: The string from which to extract the number.
    :return: The extracted number or a default value if no number is found.
    """
    match = re.search(r'(\d+)', s)
    return int(match.group(0)) if match else float('inf')

def find_files(directory, extensions):
    """
    Recursively find files with given extensions in the specified directory
    and sort them based on numerical order extracted from filenames and folder names.
    
    :param directory: The root directory to search in.
    :param extensions: A tuple of file extensions to search for.
    :return: A list of tuples where each tuple contains the file path, the folder name, and the file name.
    """
    files_list = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        # Sort the directories numerically
        dirs.sort(key=extract_number_from_string)
        
        # Filter and sort files numerically within each folder
        sorted_files = sorted(
            (file for file in files if file.endswith(extensions)),
            key=extract_last_three_digits
        )
        
        for file in sorted_files:
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            files_list.append((file_path, folder_name, file))
    
    return files_list

def save_to_csv(data, output_file):
    """
    Save the list of tuples to a CSV file.
    
    :param data: List of tuples containing file path, folder name, and file name.
    :param output_file: The output CSV file path.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Path', 'Folder Name', 'File Name'])
        writer.writerows(data)

# Define the directory to search and the file extensions
directory_to_search = "UR Fall Detection Dataset"
file_extensions = (".mp4", ".jpeg", ".png")

# Find all files with the specified extensions
found_files = find_files(directory_to_search, file_extensions)

# Save the found files to a CSV file
output_csv = "found_files.csv"
save_to_csv(found_files, output_csv)

# Print the found files with their folders
countfiles = 0
for file_path, folder_name, file_name in found_files:
    print(str(countfiles) + ": "+f"File: {file_path}, Folder: {folder_name}, File Name: {file_name}")
    countfiles += 1