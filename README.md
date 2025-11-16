
# 💡 Project Overview & Access

This application classifies images into five categories: Glass, Metal, Paper, Plastic, and Trash. It uses Transfer Learning with MobileNetV2 for high accuracy and is deployed as a live web application.
Live Application URL:https://huggingface.co/spaces/pranav0909/waste-classifier
Model Architecture: MobileNetV2 (Frozen Feature Extractor) with a custom Dense Classifier Head.
Final Test Accuracy: 79% approx

# 🗓️ Development Sprint Log

 Week 1: Data Setup and Baseline ModelThe focus was on preparing the project foundation. We collected and organized the dataset, implemented the ImageDataGenerator for loading and pixel rescaling, and trained a baseline custom CNN. The diagnostic step confirmed the initial 63% accuracy model was overfitting, defining the need for a stronger approach in the following week.

 Week 2: Transfer Learning and Optimization (Final Model)This week was dedicated to performance. We integrated MobileNetV2 for Transfer Learning, freezing the pre-trained weights and adding a custom classifier head. We introduced robust regularization, combining Data Augmentation (rotation, zoom) with L2 Regularization on the dense layers. The model was trained using EarlyStopping and ReduceLROnPlateau callbacks, successfully pushing the final test accuracy to approx 79%. The final model was saved as waste_cnn_mobilenet.keras.

 Week 3: Interface and Deployment
The project shifted to user accessibility and delivery. We built the Gradio interface using the app.py script and corrected the critical class index mismatch error. For deployment, we configured Git LFS to handle the large model file and pushed all code and dependencies to the GitHub repository. The project was successfully deployed by linking the repository to a Hugging Face Space, making the application live.

🔧 Prerequisites and Local Run
To run this project locally, ensure you have the repository cloned and execute the following:

# Install required Python packages
pip install -r requirements.txt

# Run the Gradio app locally
python app.py
