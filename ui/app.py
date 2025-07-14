from flask import Flask, request, jsonify, render_template
import base64
import os
import joblib
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing import image


# Initialize Flask app
app = Flask(__name__)

try:
    emotion_model = load_model(
        'models/efficinetNet_model_6743.keras'
    )

    gender_model = load_model(
        'models/gender_model.keras'
    )

    age_model = load_model(
        'models/age_model.keras'
    )

    ethnicity_model = load_model(
        'models/ethnicity_model.keras'
    )

    print("Models loaded successfully with custom objects.")
except Exception as e:
    print(f"Failed to load model: {e}")


def preprocess_image(image, target_size=(224, 224), to_grayscale=False):
    """Preprocess the image to match the model input shape."""
    # Convert the image to RGB first to ensure consistent processing
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to the specified target size
    img = image.resize(target_size)
    
    # Convert to grayscale if needed
    if to_grayscale:
        img = img.convert('L')  # Convert to grayscale ('L' mode)
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    else:
        img = np.array(img) / 255.0  # Normalize pixel values
    
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load the scaler used during training
scaler_path = '/Users/shwetakakade/Documents/Dissertation/scaler.pkl'

scaler = joblib.load(scaler_path)
print(f"Scaler loaded from {scaler_path}")


@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' exists in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction request."""
    try:
        # Extract and decode the base64 image from the request JSON
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image'].split(",")[1]  # Extract base64 part of the image
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Preprocess images for the models
        # Preprocess for emotion model (input size 224x224, keep RGB)
        preprocessed_emotion_image = preprocess_image(image, target_size=(224, 224), to_grayscale=False)
        print(f"Preprocessed emotion image shape: {preprocessed_emotion_image.shape}")

        # Preprocess for age_gender model (input size 224x224, convert to grayscale)
        preprocessed_age_gender_image = preprocess_image(image, target_size=(224, 224), to_grayscale=True)
        print(f"Preprocessed age_gender image shape: {preprocessed_age_gender_image.shape}")

        # Predict emotion
        emotion_prediction = emotion_model.predict(preprocessed_emotion_image)
        print(f"Emotion prediction: {emotion_prediction}")
        predicted_emotion = decode_emotion(emotion_prediction)
        print(f"Decoded emotion: {predicted_emotion}")

        # Predict gender using the separate gender model
        gender_prediction = gender_model.predict(preprocessed_age_gender_image)
        print(f"Gender prediction: {gender_prediction}")
        predicted_gender = decode_gender(gender_prediction)
        print(f"Gender prediction: {predicted_gender}")

        age_prediction = age_model.predict(preprocessed_age_gender_image)
        print(f"Age prediction: {age_prediction}")
        predicted_age = decode_age(age_prediction)
        print(f"Age prediction: {predicted_age}")

        # Predict ethnicity using the separate ethnicity model
        ethnicity_prediction = ethnicity_model.predict(preprocessed_age_gender_image)
        predicted_ethnicity = decode_ethnicity(ethnicity_prediction)
        print(f"Ethnicity prediction: {predicted_ethnicity}")

        # Create the response dictionary
        response = {
            'emotion': predicted_emotion,
            'age': predicted_age,
            'gender': predicted_gender,
            'ethnicity': predicted_ethnicity
        }

        return jsonify(response)

    except KeyError as e:
        # Handle cases where expected data keys are missing
        return jsonify({'error': f'Missing key in request data: {str(e)}'}), 400
    except Exception as e:
        # Catch-all for any other exceptions
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


def preprocess_image(image, target_size, to_grayscale):
    if to_grayscale:
        image = image.convert('L')  # Convert to grayscale
    image = image.resize(target_size)
    return np.expand_dims(np.array(image).astype(np.float32), axis=0)  # Add batch dimension



def decode_emotion(prediction):
    emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Ensure prediction is a 2D array with shape (1, 7)
    prediction = np.array(prediction)
    
    # Get the index of the highest probability
    emotion_index = np.argmax(prediction)
    
    # Get the emotion label
    emotion = emotions[emotion_index]
    
    # Get the probability of each emotion
    probabilities = prediction[0]
    
    # Create a formatted response
    response = f"Emotion prediction: "
    response += ', '.join(f"{emotions[i]} {probabilities[i]:.2f}" for i in range(len(emotions)))
    
    return response

def decode_gender(prediction):
    """Decode gender prediction to a human-readable label."""
    gender_index = int(prediction)  # Assuming gender is binary, 0 for Female, 1 for Male
    if gender_index >= 1:
        return 'Female'
    else:
        return 'Male'


def decode_age(prediction):

    """Decode age prediction to a human-readable format within the range 0-120.""" 
    return int(round(prediction[0][0]/100))

def decode_ethnicity(prediction):
    """Decode ethnicity prediction to a human-readable label."""
    ethnicities = ['White', 'Black', 'Asian', 'Indian', 'Others']  # Replace with actual labels
    ethnicity_index = int(np.argmax(prediction))
    return ethnicities[ethnicity_index]


if __name__ == '__main__':
    app.run(debug=True)
