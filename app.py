import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. Load the Model and Setup ---
# *** IMPORTANT: UPDATE THE MODEL PATH ***
MODEL_PATH = "waste_cnn_mobilenet.keras" 
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the file 'waste_cnn_mobilenet.keras' is in this directory.")
    exit()

# *** IMPORTANT: USE THE CONFIRMED CLASS ORDER ***
# The order must match: {'glass': 0, 'metal': 1, 'paper': 2, 'plastic': 3, 'trash': 4}
CLASS_NAMES = ['glass', 'metal', 'paper', 'plastic', 'trash'] 

IMAGE_SIZE = (128, 128) 

# --- 2. Prediction Function ---
def classify_waste(image):
    """
    Preprocesses the input image and makes a prediction using the CNN model.
    """
    if image is None:
        # Return zeros if no image is uploaded
        return {name: 0.0 for name in CLASS_NAMES}

    # 1. Resize and Convert (PIL to NumPy array)
    # The model expects (128, 128)
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype='float32')

    # 2. Rescale (Must match the 1./255 scaling used during training)
    img_array = img_array / 255.0

    # 3. Expand dimensions to create a batch of 1
    # Shape changes from (128, 128, 3) to (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0) 

    # 4. Predict
    predictions = model.predict(img_array, verbose=0)[0] # verbose=0 suppresses Keras output during prediction
    
    # 5. Format Output for Gradio (Map confidence scores to class names)
    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# --- 3. Build and Launch Gradio Interface ---
# Define the interface components
image_input = gr.Image(type="pil", label="Upload Waste Image")
# Show all 5 classes in the output label
label_output = gr.Label(num_top_classes=len(CLASS_NAMES)) 

# Define the interface
interface = gr.Interface(
    fn=classify_waste,
    inputs=image_input,
    outputs=label_output,
    title="♻️ Waste Classification App",
    description="Upload a waste item image to get a prediction from the highly accurate MobileNetV2 model.",
    # You can add a path to an example image here for users to quickly test
    # examples=['path/to/your/example_plastic.jpg'] 
)

# Launch the interface
print("\nLaunching Gradio interface...")
interface.launch(inbrowser=True)

