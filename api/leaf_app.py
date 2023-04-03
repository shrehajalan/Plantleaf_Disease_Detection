from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import pickle
import cv2



app=FastAPI()
with open('../goodpotato_model_final.pkl','rb')as f:
    MODEL_potato=pickle.load(f)

with open('../goodtomato_model_final.pkl','rb')as f:
    MODEL_tomato=pickle.load(f)

with open('../goodcorn_model_final.pkl','rb')as f:
    MODEL_corn=pickle.load(f)

CLASS_NAMES_potato = ["Healthy", "Early Blight", "Late Blight"]

CLASS_NAMES_tomato= ["Healthy", "Early Blight", "Late Blight","bacterial_spot","Septoria_spot","Leaf_Mold","Spider_mites Two-spotted_spider_mite","Target_Spot","Tomato_mosaic_virus","Tomato_Yellow_Leaf_Curl_Virus"]

CLASS_NAMES_corn = ["Healthy", "Blight", "Common rust"]

@app.get("/ping")

async def ping():
    return "Hello,I am alive"

def read_file_as_image(data) -> np.ndarray:
    #image = np.array(Image.open(BytesIO(data))
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    k=cv2.resize(img,(256,256))
    k1=np.array(k)
    return k1

@app.post("/predict_potato")

async def predict_potato(file: UploadFile = File(...)):
    image= read_file_as_image(await file.read())
    ko=image/255
    print(ko)
    img_batch = np.expand_dims(ko, 0)
    predictions = MODEL_potato.predict(img_batch)
    print(predictions)
    predicted_class = CLASS_NAMES_potato[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'name' :'Potato Leaf',
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.post("/predict_tomato")
async def predict_tomato(file: UploadFile = File(...)):
    image= read_file_as_image(await file.read())
    ko=image/255
    print(ko)
    img_batch = np.expand_dims(ko, 0)
    predictions = MODEL_tomato.predict(img_batch)
    print(predictions)
    predicted_class = CLASS_NAMES_tomato[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'name':"Tomato Leaf",
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.post("/predict_corn")

async def predict_corn(file: UploadFile = File(...)):
    image= read_file_as_image(await file.read())
    ko=image/255
    print(ko)
    img_batch = np.expand_dims(ko, 0)
    predictions = MODEL_corn.predict(img_batch)
    print(predictions)
    predicted_class = CLASS_NAMES_corn[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'name' :'corn Leaf',
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)