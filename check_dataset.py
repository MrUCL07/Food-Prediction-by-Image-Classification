import os


train_path = r"D:\Rithvik\College\FoodImageProcessor\FoodDatabase\train"
test_path = r"D:\Rithvik\College\FoodImageProcessor\FoodDatabase\test"


print("\n========== TRAINING DATA ==========\n")

for folder in sorted(os.listdir(train_path)):

    folder_path = os.path.join(
        train_path,
        folder
    )

    if os.path.isdir(folder_path):

        images = os.listdir(folder_path)

        print(
            folder,
            "->",
            len(images),
            "images"
        )


print("\n========== TEST DATA ==========\n")

for folder in sorted(os.listdir(test_path)):

    folder_path = os.path.join(
        test_path,
        folder
    )

    if os.path.isdir(folder_path):

        images = os.listdir(folder_path)

        print(
            folder,
            "->",
            len(images),
            "images"
        )