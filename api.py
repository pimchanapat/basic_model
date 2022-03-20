import tensorflow as tf
load_model = tf.keras.models.load_model

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
import secrets
import os
from dotenv import load_dotenv

load_dotenv(os.path.join('.env'))

API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Incorrect username or password',
            headers={'WWW-Authenticate': 'Basic'},
        )
    return credentials.username

app = FastAPI()

class Data(BaseModel):
    x:float
    y:float

def loadModel():
    global predict_model

    predict_model = load_model('model1.h5')

loadModel()

async def predict(data):
    classNameCat = {0:'class_A', 1:'class_B', 2:'class_C'}
    X = np.array([[data.x, data.y]])

    pred = predict_model.predict(X)

    res = np.argmax(pred, axis=1)[0]
    category = classNameCat[res]
    confidence = float(pred[0][res])
        
    return category, confidence

@app.post('/getclass/')
async def get_class(data: Data, username: str = Depends(get_current_username)):
    category, confidence = await predict(data)
    res = {'class': category, 'confidence':confidence}
    return {'results': res}