# Import libraries
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# Apply the asyncio patch
nest_asyncio.apply()

# Define FastAPI app
app = FastAPI()

# Load your trained model
model = load_model('./model111.h5')

# Class labels for your model (replace with your actual labels)
class_labels = {0: 'Apple', 1: 'Banana', 2: 'Orange', 3: 'Strawberry', 4: 'Grapes'}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def predict(img_file: UploadFile = File(...)):
    # Read the image file
    img = Image.open(img_file.file).convert('RGB')
    img = img.resize((150, 150))  # Resize image to match model input

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels.get(predicted_class, "Unknown")

    return {"prediction": predicted_label}
