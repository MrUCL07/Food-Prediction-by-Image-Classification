import os
import numpy as np
from datasets import load_dataset

dataset_path = r"D:\Rithvik\College\FoodImageProcessor\IndianFoodDataset"

output_path = r"D:\Rithvik\College\FoodImageProcessor\FoodDatabase"

train_file = os.path.join(
    dataset_path,
    "data",
    "train-00000-of-00002-5de00600cb70f0ee.parquet"
)

train_file2 = os.path.join(
    dataset_path,
    "data",
    "train-00001-of-00002-7cca688335b50fc2.parquet"
)

test_file = os.path.join(
    dataset_path,
    "data",
    "test-00000-of-00001-75236f054535aed8.parquet"
)


print("Loading training dataset...")

train_dataset = load_dataset(
    "parquet",
    data_files=[train_file, train_file2],
    split="train"
)

print("Loading test dataset...")

test_dataset = load_dataset(
    "parquet",
    data_files=test_file,
    split="train"
)

class_names = [
    "biryani",
    "cholebhature",
    "dabeli",
    "dal",
    "dhokla",
    "dosa",
    "jalebi",
    "kathiroll",
    "kofta",
    "naan",
    "pakora",
    "paneer",
    "panipuri",
    "pavbhaji",
    "vadapav"
]

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "=", name)

train_output = os.path.join(output_path, "train")
test_output = os.path.join(output_path, "test")

os.makedirs(train_output, exist_ok=True)
os.makedirs(test_output, exist_ok=True)

for class_name in class_names:

    os.makedirs(
        os.path.join(train_output, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(test_output, class_name),
        exist_ok=True
    )

print("\nSaving training images...")

for i, item in enumerate(train_dataset):

    image = item["image"]
    label = item["label"]

    class_name = class_names[label]

    image_array = np.array(image)

    filename = os.path.join(
        train_output,
        class_name,
        f"train_{i:05d}.jpg"
    )

    from PIL import Image

    Image.fromarray(image_array).convert("RGB").save(filename)

    if (i + 1) % 500 == 0:
        print(f"Saved {i + 1} training images")

print("\nSaving test images...")

for i, item in enumerate(test_dataset):

    image = item["image"]
    label = item["label"]

    class_name = class_names[label]

    image_array = np.array(image)

    filename = os.path.join(
        test_output,
        class_name,
        f"test_{i:05d}.jpg"
    )

    from PIL import Image

    Image.fromarray(image_array).convert("RGB").save(filename)

    if (i + 1) % 200 == 0:
        print(f"Saved {i + 1} test images")


print("\n================================")
print("Dataset preparation completed!")
print("================================")
print("\nDataset location:")
print(output_path)