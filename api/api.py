from typing import Union, List
from fastapi import FastAPI, Query, UploadFile, File
import shutil
from pathlib import Path
import utils
from fastapi.middleware.cors import CORSMiddleware
from cors_origins import Origins
import torch
from fastapi.responses import FileResponse

app = FastAPI()

origins = Origins.origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_filename = ""

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    path = Path(__file__).parents[1] / "saved_images" / file.filename
    try:
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print("Error: ", e)
    global current_filename
    current_filename = file.filename
    return {"filename": file.filename}

@app.get("/predict/")
async def predict(filename: str):
    in_path = Path(__file__).parents[1] / "saved_images" / filename
    out_path = Path(__file__).parents[1] / "saved_images" / "sr.jpg"
    # print(type(str(path)))
    in_path = str(in_path)
    out_path = str(out_path)
    utils.predict_one_image(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        model_arch_name="rrdbnet_x4",
        model_weights_path="/Users/0x4ry4n/Desktop/Dev/unblurai-new/pretrained-models/ESRGAN_x4-DFO2K-25393df7.pth.tar",
        inputs_path=in_path,
        output_path=out_path
    )
    # utils.write_img_after_reading(in_path, out_path)
    return {
        "STATUS": "SUCCESS",
    }

@app.get("/image")
async def get_image():
    filename = Path(__file__).parents[1] / "saved_images" / "sr.jpg"
    return FileResponse(filename)

@app.get("/")
def read_root():
    return {"Hello": "World"}