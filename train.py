import tensorflow as tf
import numpy as np
import os
import glob


# =========================================================
# 1. DATASET PATH
# =========================================================

dataset_path = r"D:\Rithvik\College\FoodImageProcessor\FoodDatabase\train"


# =========================================================
# 2. GET CLASS NAMES
# =========================================================

class_names = sorted([
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(i, "=", class_name)

print("\nNumber of classes:", len(class_names))


# =========================================================
# 3. COLLECT IMAGE FILES
# =========================================================

image_files = []
labels = []

for label, class_name in enumerate(class_names):

    class_path = os.path.join(
        dataset_path,
        class_name
    )

    files = glob.glob(
        os.path.join(class_path, "*.jpg")
    )

    for file in files:

        image_files.append(file)
        labels.append(label)


print("\nTotal images:", len(image_files))


# =========================================================
# 4. CONVERT TO NUMPY
# =========================================================

image_files = np.array(image_files)
labels = np.array(labels)


# =========================================================
# 5. SHUFFLE DATA
# =========================================================

np.random.seed(42)

indices = np.arange(len(image_files))

np.random.shuffle(indices)

image_files = image_files[indices]
labels = labels[indices]


# =========================================================
# 6. TRAIN / VALIDATION SPLIT
# =========================================================

split = int(0.8 * len(image_files))

train_files = image_files[:split]
train_labels = labels[:split]

val_files = image_files[split:]
val_labels = labels[split:]


print("\nTraining images:", len(train_files))
print("Validation images:", len(val_files))


# =========================================================
# 7. IMAGE LOADING
# =========================================================

IMG_SIZE = 128


def load_image(filename, label):

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

    return image, label


# =========================================================
# 8. CREATE DATASETS
# =========================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_files, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_files, val_labels)
)


train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_dataset = train_dataset.shuffle(1000)

train_dataset = train_dataset.batch(32)

val_dataset = val_dataset.batch(32)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# 9. DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.05
    )

])


# =========================================================
# 10. CNN MODEL
# =========================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    # -------------------------
    # Data Augmentation
    # -------------------------

    data_augmentation,


    # -------------------------
    # Convolution Block 1
    # -------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # -------------------------
    # Convolution Block 2
    # -------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # -------------------------
    # Convolution Block 3
    # -------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # -------------------------
    # Convolution Block 4
    # -------------------------

    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # -------------------------
    # Fully Connected Layer
    # -------------------------

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),


    # -------------------------
    # Output
    # -------------------------

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# =========================================================
# 11. COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# =========================================================
# 12. MODEL SUMMARY
# =========================================================

model.summary()


# =========================================================
# 13. CALLBACKS
# =========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=7,

    restore_best_weights=True,

    mode="max"
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "indian_food_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"
)


# =========================================================
# 14. TRAIN
# =========================================================

print("\n================================")
print("STARTING TRAINING")
print("================================\n")


history = model.fit(

    train_dataset,

    epochs=40,

    validation_data=val_dataset,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]

)


# =========================================================
# 15. RESULTS
# =========================================================

print("\n================================")
print("TRAINING COMPLETED")
print("================================")


print(
    "\nBest validation accuracy:",
    max(history.history["val_accuracy"])
)


print(
    "Best training accuracy:",
    max(history.history["accuracy"])
)


print(
    "\nModel saved as:",
    "indian_food_model.keras"
)