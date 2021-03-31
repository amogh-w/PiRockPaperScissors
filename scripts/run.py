import os
import numpy as np
import base64

from pydantic import BaseModel
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import tensorflow as tf
from tensorflow.keras.models import load_model


class Data(BaseModel):
    image: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"],
)

#  model = load_model("./rps-cnn.h5")


@app.post("/predict")
def predict(data: Data):
    data_dict = data.dict()

    with open("imageToSave.jpeg", "wb") as fh:
        fh.write(base64.decodebytes(bytes(data_dict["image"][23:], "utf-8")))

    #  image = tf.keras.preprocessing.image.load_img(
    #      "imageToSave.jpeg", target_size=(300, 200)
    #  )
    #  input_arr = tf.keras.preprocessing.image.img_to_array(image)
    #  input_arr = np.array([input_arr])
    #  predictions = model.predict(input_arr)

    return {"prediction": str("lol")}
