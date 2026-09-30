import os
from PIL import Image

dataset_path = "Database"

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nChecking: {folder}")

    count = 0

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        try:
            img = Image.open(file_path)
            img.verify()
            count += 1

        except:
            print("Invalid image:", file_path)

    print("Valid images:", count)