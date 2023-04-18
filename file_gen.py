import os
import csv

# Directory path containing the files
directory = '/Users/samridhiRoshan/Desktop/rough/ocr_gen_images'

# List to store the old and new filenames
filename_list = []
new_name = 199

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):  # Change the file extension as needed
        old_filename = os.path.join(directory, filename)
        new_filename = os.path.join(directory, str(new_name) + '.jpg')  # Change the new filename format as needed
        os.rename(old_filename, new_filename)
        toAdd = filename[:-5]  #'.jpg' 
        filename_list.append((toAdd, str(new_name) + '.jpg'))
        new_name = new_name + 1

# Write the filenames to a CSV file
csv_filename = 'filename_mapping.csv'  # Change the CSV filename as needed
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['old_filename', 'new_filename']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for filenames in filename_list:
        writer.writerow(filenames)

print('Filenames changed and CSV file generated successfully.')
