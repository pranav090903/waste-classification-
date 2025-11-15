import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.applications import MobileNetV2 # New import for Transfer Learning
import matplotlib.pyplot as plt

# ============================
# 1. Set dataset path
# ============================
# NOTE: Ensure these paths are correct on your system.
train_dir = r"C:\Users\ASUS\Downloads\waste-classification-\waste_classification_processed\train"
test_dir = r"C:\Users\ASUS\Downloads\waste-classification-\waste_classification_processed\test"

# ============================
# 2. Load dataset using ImageDataGenerator (With Augmentation)
# ============================
image_size = (128, 128) # Must match the input size for MobileNetV2
batch_size = 32

# Data Augmentation for Training Data (Crucial for generalization)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only Rescaling for Test Data
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# PRINT THE CLASS INDICES HERE TO FIX GRADIO LABELING
print("Class Indices:", train_data.class_indices) 

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ============================
# 3. Build CNN Model (Transfer Learning with MobileNetV2) 🚀
# ============================

# 1. Load the Pre-trained Base Model (MobileNetV2)
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False, # We don't need the classification layer from ImageNet
    weights='imagenet' 
)

# Freeze the base layers so the pre-trained features are preserved
base_model.trainable = False 

# 2. Build the New Classifier Head
model = models.Sequential([
    base_model, # The frozen feature extractor
    
    layers.GlobalAveragePooling2D(), # Efficiently reduces the feature map dimensions
    
    layers.Dense(128, 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)), # Added L2 Regularization
    
    layers.Dropout(0.5), # Standard Dropout 
    
    # Final output layer with the correct number of classes
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================
# 4. Train model (With Robust Callbacks)
# ============================
epochs = 50 

# Define callbacks for better training control
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8, # Increased patience slightly for a stronger model
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data,
    callbacks=[early_stop, reduce_lr]
)

# ============================
# 5. Save model
# ============================
# NOTE: Saving in the native Keras format is recommended
model.save("waste_cnn_mobilenet.keras")
print("Model saved as waste_cnn_mobilenet.keras")

# ============================
# 6. Plot accuracy & loss
# ============================

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title("Loss")
plt.show()