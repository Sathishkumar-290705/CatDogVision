import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
# ── 1. PATHS ──────────────────────────────────────────────────────────────────
TRAIN_DIR = r"E:\PROJECTS\FIND CAT OR DOG\Data\train"
VAL_DIR   = r"E:\PROJECTS\FIND CAT OR DOG\Data\validation"
IMG_SIZE  = (150, 150)   # resize every image to 150×150
BATCH     = 32
EPOCHS    = 20

# ── 2. DATA AUGMENTATION ──────────────────────────────────────────────────────
# Only applied to TRAINING data — val data is never augmented

train_datagen = ImageDataGenerator(
    rescale           = 1.0 / 255,   
    rotation_range    = 40,          # randomly rotate image up to 40 degrees
    width_shift_range = 0.2,         # randomly shift image left/right by 20%
    height_shift_range= 0.2,         # randomly shift image up/down by 20%
    shear_range       = 0.2,         # tilt/slant the image
    zoom_range        = 0.2,         # randomly zoom in
    horizontal_flip   = True,        # randomly flip image left↔right
    fill_mode         = "nearest"    # fill empty pixels after shift/rotate
)

# Validation — only normalize, NO augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)


# ── 3. LOAD IMAGES FROM FOLDERS ───────────────────────────────────────────────
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_SIZE,    # resize all images to 150×150
    batch_size  = BATCH,
    class_mode  = "binary"     # binary = 2 classes (dog=1, cat=0)
)


val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH,
    class_mode  = "binary"
)


# ── 4. BUILD THE CNN MODEL ────────────────────────────────────────────────────
model = models.Sequential([
    # --- Conv Block 1 ---
    # 32 kernels of 3×3 sliding on 150×150×3 image → 148×148×32
    
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    # Max Pooling shrinks 148×148 → 74×74
    layers.MaxPooling2D(2, 2),

    # --- Conv Block 2 ---
    # 64 kernels → 72×72×64
    layers.Conv2D(64, (3, 3), activation="relu"),
    # Shrinks 72×72 → 36×36
    layers.MaxPooling2D(2, 2),

    # --- Conv Block 3 ---
    # 128 kernels → 34×34×128
    layers.Conv2D(128, (3, 3), activation="relu"),
    # Shrinks 34×34 → 17×17
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(256,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),


    # --- Flatten ---
    # 17×17×128 = 36,992 values → 1D vector
    layers.Flatten(),

    # --- Fully Connected ---
    layers.Dense(512, activation="relu"),   # 36992 → 512 neurons
    layers.Dropout(0.5),                    # randomly turn off 50% neurons (prevents overfitting)

    # --- Output ---
    # 1 neuron: outputs 0.0 (cat) to 1.0 (dog)
    layers.Dense(1, activation="sigmoid")
])

print("model summary")
model.summary()

# ── 5. COMPILE ────────────────────────────────────────────────────────────────
model.compile(
    optimizer = "adam",                  # adjusts learning rate automatically
    loss      = "binary_crossentropy",   # loss function for 2-class problem
    metrics   = ["accuracy"]
)

# ── 6. TRAIN ──────────────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs          = EPOCHS,
    validation_data = val_generator
)

# ── 7. SAVE MODEL ─────────────────────────────────────────────────────────────
model.save("dog_cat_model.h5")
print("Model saved as dog_cat_model.h5")

# ── 8. PREDICT A NEW IMAGE ────────────────────────────────────────────────────


# def predict(img_path):
#     img = image.load_img(img_path, target_size=IMG_SIZE)       # load image
#     img_array = image.img_to_array(img) / 255.0                # normalize
#     img_array = np.expand_dims(img_array, axis=0)              # add batch dim → (1,150,150,3)

#     prediction = model.predict(img_array)[0][0]                # get output value
#     print("Prediction accuracy ",prediction)
#     if prediction > 0.5:
#         print(f"Dog  (confidence: {prediction:.2f})")
#     else:
#         print(f"Cat  (confidence: {1 - prediction:.2f})")

# # predict("test_images/my_dog.jpg")   ← uncomment and use your image path
