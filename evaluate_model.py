import tensorflow as tf
import numpy as np
import os
import glob


# =========================================================
# 1. PATHS
# =========================================================

test_path = r"D:\Rithvik\College\FoodImageProcessor\FoodDatabase\test"

model_path = r"D:\Rithvik\College\FoodImageProcessor\indian_food_model.keras"


# =========================================================
# 2. LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(model_path)

print("\nModel loaded successfully.")


# =========================================================
# 3. GET CLASS NAMES
# =========================================================

class_names = sorted([

    folder

    for folder in os.listdir(test_path)

    if os.path.isdir(
        os.path.join(test_path, folder)
    )

])


print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(i, "=", class_name)


# =========================================================
# 4. LOAD TEST IMAGES
# =========================================================

image_files = []
labels = []


for label, class_name in enumerate(class_names):

    class_path = os.path.join(
        test_path,
        class_name
    )

    files = glob.glob(
        os.path.join(class_path, "*.jpg")
    )

    for file in files:

        image_files.append(file)

        labels.append(label)


print("\nTotal test images:", len(image_files))


# =========================================================
# 5. LOAD AND PREPROCESS IMAGE
# =========================================================

IMG_SIZE = 128


def load_image(filename):

    image = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


# =========================================================
# 6. TEST MODEL
# =========================================================

correct = 0
total = len(image_files)


print("\nStarting evaluation...\n")


for i in range(total):

    image = load_image(
        image_files[i]
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_label = np.argmax(
        prediction[0]
    )

    actual_label = labels[i]


    if predicted_label == actual_label:

        correct += 1


    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{total}"
        )


# =========================================================
# 7. FINAL ACCURACY
# =========================================================

accuracy = correct / total


print("\n================================")
print("FINAL TEST RESULTS")
print("================================")

print(
    "\nCorrect predictions:",
    correct
)

print(
    "Total images:",
    total
)

print(
    "Test accuracy:",
    accuracy * 100,
    "%"
)