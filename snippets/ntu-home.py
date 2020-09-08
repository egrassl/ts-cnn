
from os import listdir
from os.path import isfile, join
from shutil import copyfile

#*********** CONFIGURATION ***********

# Path where the full dataset is located
dataset_path = r"D:\Mestrado\code\databases\NTU\nturgb+d_rgb"
# Path where the new dataset will be created
new_dataset_path = r"D:\Mestrado\code\databases\NTU-HOME\raw"
# Classes that will be used in the new dataset
classes = ["A001", "A002", "A003", "A008", "A009", "A011", "A012", "A027", "A028", "A032", "A037", "A041", "A043",
           "A044", "A045", "A046", "A047", "A048", "A069", "A070", "A074", "A085", "A103", "A104", "A105"]

#*********** EXECUTION ***********

print(len(listdir(dataset_path)))

# Get's file to copy from source location
samples = [s for s in listdir(dataset_path) if isfile(join(dataset_path, s)) and any(c in s for c in classes)]

print(len(samples))

# Copy these files to the new dataset destination
for s in samples:
    copyfile(join(dataset_path, s), join(new_dataset_path, s))