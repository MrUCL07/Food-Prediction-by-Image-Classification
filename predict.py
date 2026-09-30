import tensorflow as tf
import numpy as np
import os


# ============================================================
# CLASS NAMES
# ============================================================

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


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    "indian_food_model.keras"
)


print("\nModel loaded successfully.")


# ============================================================
# ASK USER FOR IMAGE
# ============================================================

image_path = input(
    "\nEnter the path of the food image: "
)


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):

    print("\nImage not found!")

    exit()


print("\nImage found:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)


# ============================================================
# CONVERT IMAGE TO NUMPY ARRAY
# ============================================================

image_array = tf.keras.utils.img_to_array(
    image
)


# ============================================================
# NORMALIZE IMAGE
# ============================================================

image_array = image_array / 255.0


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(
    image_array,
    verbose=0
)


# ============================================================
# GET PREDICTED CLASS
# ============================================================

predicted_index = np.argmax(
    prediction[0]
)


predicted_food = class_names[
    predicted_index
]


# ============================================================
# GET CONFIDENCE
# ============================================================

confidence = prediction[0][
    predicted_index
] * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print(
    "\n=========================================="
)

print(
    "           FOOD PREDICTION"
)

print(
    "=========================================="
)

print(
    "\nPredicted Food:",
    predicted_food.upper()
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)


print(
    "\n=========================================="
)


# ============================================================
# SHOW ALL CLASS PROBABILITIES
# ============================================================

print(
    "\nAll Class Probabilities:\n"
)


for i in range(len(class_names)):

    probability = prediction[0][i] * 100

    print(
        f"{class_names[i]:15} -> "
        f"{probability:.2f}%"
    )