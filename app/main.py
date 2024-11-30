# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.requests import Request
# import numpy as np
# from keras.models import load_model
# from PIL import Image, ImageOps
#
# app = FastAPI()
#
# # Mount static and templates directories
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
# templates = Jinja2Templates(directory="app/templates")
#
# # Load model and labels
# model = load_model("keras_Model.h5", compile=False)
# with open("labels.txt", "r") as f:
#     class_names = f.readlines()
#
# @app.get("/", response_class=HTMLResponse)
# async def homepage(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
#
# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, file: UploadFile = File(...)):
#     # Preprocess image
#     image = Image.open(file.file).convert("RGB")
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#     image_array = np.asarray(image)
#     normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     data[0] = normalized_image_array
#
#     # Predict
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     name = class_names[index].strip().split()
#     # confidence_score = round(prediction[0][index] * 100, 2)
#     class_name = ' '.join(name[1:])
#
#
#     # Render result page
#     return templates.TemplateResponse(
#         "result.html",
#         {
#             "request": request,
#             "prediction": class_name,
#             # "confidence": confidence_score,
#         },
#     )


from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
from keras.models import load_model

app = FastAPI()

# Mount static and templates directories
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model and labels
model = load_model("keras_Model.h5", compile=False)
with open("labels.txt", "r") as f:
    class_names = f.readlines()

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Preprocess image
        image = Image.open(file.file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        confidence_score = round(prediction[0][index] * 100, 2)
        name = class_names[index].strip().split()
        class_name = ' '.join(name[1:])
        # Render result page
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": class_name,
                "confidence": confidence_score,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": "Error processing the image.",
                "confidence": "N/A",
            },
        )


@app.post("/predict-url", response_class=HTMLResponse)
async def predict_url(request: Request, url: str = Form(...)):
    try:
        # Fetch image from URL
        response = requests.get(url)
        response.raise_for_status()  # Raise error if URL is invalid
        image = Image.open(BytesIO(response.content)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = round(prediction[0][index] * 100, 2)

        # Render result page
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": class_name,
                "confidence": confidence_score,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": "Error: Unable to fetch or process the image.",
                "confidence": "N/A",
            },
        )
