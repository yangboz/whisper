import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware


from PIL import Image
from io import BytesIO
from datetime import datetime


import base64

import json

import timeit
from yolov5 import YOLOv5
import os

import imageio

import aiofiles

import cv2


from dotenv import load_dotenv
import pymongo

from fastapi import Request
from fastapi import WebSocket

from fastapi.templating import Jinja2Templates

#from fastapi.staticfiles import StaticFiles

#import random


import time
import whisper
# from mongoengine import DateTimeField,metaField

# set model params
# model_path = "/Users/apple/Documents/yolov5pip/yolov5s.pt" # it automatically downloads yolov5s model to given path
#model_path = os.getcwd()+"/yolov5n6-1800.pt" # it automatically downloads yolov5s model to given path

device = "cpu" # or "cpu"


#load_dotenv()  # take environment variables from .env.

#MONGO_DB_URI = os.environ.get("MONGO_DETAILS")
#print("MONGO_DB_URI: ",MONGO_DB_URI)

# init yolov5 model

#yolov5 = YOLOv5(model_path, device)

app_desc = """<h2>OPenAI whisper service</h2>
<h2>Try  rest functions and verify return response </h2>
<br>by yangboz@SMKT"""

app = FastAPI(title='restful FastAPI Starter Pack', description=app_desc)

app.data = ['a','b','c','d']
#crossdomain configs
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/static",StaticFiles(directory='static'))
#templates = Jinja2Templates(directory="templates")


#with open('measurements.json', 'r') as file:
#    raw_measurements = json.loads(file.read())
#    print("raw_measurements[0]:",raw_measurements[0])
#    measurements = iter(raw_measurements)
# testing with  mongodb or direct websocket broadcasting
# client_pymongo = pymongo.MongoClient(MONGO_DB_URI)
# print("mongodb conn:",client_pymongo)
# testdb = client_pymongo.test
# print("mongodb testdb:",testdb)
# print("mongodb testdb explain:",testdb.collection.explain())


#
# testdb.adminCommand( { getParameter: 1, featureCompatibilityVersion: 1 } )
# testdb.adminCommand( { setFeatureCompatibilityVersion: "5.0" } )
# testdb.command("create", 'ebikeparkingData', timeseries={ 'timeField': 'timestamp', 'metaField': 'symbol', 'granularity': 'hours' })
# testdb.create_collection("ebikeparkingData", 
# { timeseries: { 
#                'DateTimeField': "date", 
#                'metaField': "symbol",
#                'granularity': "minutes" },'expireAfterSeconds': 9000 })
# # #
# mycol = testdb["ebikeparkingData"]

# print("ebikeparkingData col:",mycol)
# print("ebikeparkingData col descibe:",mycol.explain())

# Getting the current date and time
#dt = datetime.now()
# getting the timestamp
# ts = datetime.timestamp(dt)

#print("Date and time is:", dt)
#print("Timestamp is:", ts)

# current timestamp
x = time.time()
print("[time]Timestamp:", x)

#mydict = { "ebike": 2, "person": 1,"sensorId":0,"ts":  ts}
# mydict = { "ebike": 2, "person": 1,"sensorId":0}
#print("mydict:",mydict)
raw_measurements.append(mydict)
#print("raw_measurements[-1]:",raw_measurements[-1])
# x = mycol.insert_one(mydict)
# print(x.inserted_id)
# print(mycol.findOne())




#utilize functions

def get_measurements()->iter:
    return  iter(raw_measurements)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def savedata2db(jsonstr) -> str: # return inserted_id as 6264f6974147c94fbc0936e6
    mycol = testdb["ebikeparkingData"]
    x = col.insert_one(jsonstr)
    print("insert_one:",x.inserted_id)
    return x.inserted_id


# @app.get("/")
# def read_root(request: Request):
#     return templates.TemplateResponse("index.htm", {"request": request})

@app.get("/api/")
def get_data():
    return app.data

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.htm", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await asyncio.sleep(0.1)
        payload = next(measurements)
        await websocket.send_json(payload)
        print("sent2 ws payload:",payload)

@app.post("/detect/video")
async def detect_in_video(file: UploadFile = File(...),max_frame_number:int=10):
    extension = file.filename.split(".")[-1] in ("mp4", "mpeg", "avi")
    if not extension:
        return "Video must be mp4 or mpeg/avi format!"
    # img_uploaded = read_imagefile(await file.read())
    # img_data = None
    # if type(file) is str :
    #     imgdata = base64.b64decode(imgstr)
    #     print("imgdata:",imgdata)
    # # filename = '%s.jpg' % photo_name 
    # with open(file.filename, 'wb') as f:
    #     f.write(imgdata)
    # img_uploaded = read_imagefile(await file.read())

    # assert img_uploaded is None
    # print("file.filename:",file.filename)
    # print("img_uploaded:",img_uploaded)
    img_uploaded_path = "uploaded_"+ str(file.filename)
    print("img_uploaded_path:",img_uploaded_path)
    # img_uploaded.save(img_uploaded_path)
    async with aiofiles.open(img_uploaded_path, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    #split video to imgs.
    print(os.getcwd())
    abs_filepath = os.getcwd()+"/"+img_uploaded_path
    print("abs_filepath:",abs_filepath)
    # reader = imageio.get_reader(abs_filepath)
    #split video to imgs

    vidcap = cv2.VideoCapture(abs_filepath)
    imgframes = []
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(f'splited/'+"frame%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      imgframes.append(image)

      count += 1
      if count%max_frame_number==0:
        break

    #start = datetime.now()
    #opt = detector.parse_opt()
    #opt = Namespace(weights='yolov5n6.pt', source=img_uploaded_path)
    #obj = detector.run(**vars(opt))
   # perform inference with larger input size
#results = yolov5.predict(image1, size=1280, augment=True)
    # perform inference with test time augmentation
    print("imgframes:",imgframes)
    results = yolov5.predict(imgframes,size=1280, augment=True)
    print("detected obj:",results)
    column = results.pandas().xyxy[0].name
    print("records:",results.pandas().xyxy[0].to_json(orient="records"))
    # obj = {"person":0,"tie":0}
    jsonstr = column.value_counts().to_json()
    print("column:",column,"value_count:",column.value_counts(),jsonstr)
    jsonobj = json.loads(jsonstr)
    print("detected objson:",jsonobj)
        # obj = DeepFace.analyze(img_path = img_uploaded_path, actions = ['age', 'gender', 'race', 'emotion'],enforce_detection=False)
    #end = datetime.now()
    exect = round(timeit.timeit(lambda:results),3)*1000
    print("The time of execution of above program is :",exect,"ms")
    jsonobj['execT'] = str(exect)+ "ms"
    print("detected response:",jsonobj)
    return jsonobj

@app.post("/detect/voice")
async def detect_in_voice(file: UploadFile = File(...),language:str='',model:str='small'):
    extension = file.filename.split(".")[-1] in ("wav", "mp3", "flac")
    if not extension:
        return "voice must be acceptable format!"
    voice_uploaded = read_imagefile(await file.read())
    # img_data = None
    # if type(file) is str :
    #     imgdata = base64.b64decode(imgstr)
    #     print("imgdata:",imgdata)
    # # filename = '%s.jpg' % photo_name 
    # with open(file.filename, 'wb') as f:
    #     f.write(imgdata)
    # img_uploaded = read_imagefile(await file.read())

    # assert img_uploaded is None
    # print("file.filename:",file.filename)
    # print("img_uploaded:",img_uploaded)
    voice_uploaded_path = "uploaded_"+ str(file.filename)
    print("voice_uploaded_path:",img_uploaded_path)
    img_uploaded.save(img_uploaded_path)
model = whisper.load_model(model)
result = model.transcribe()
print(result["text"])
    #start = datetime.now()
    #opt = detector.parse_opt()
    #opt = Namespace(weights='yolov5n6.pt', source=img_uploaded_path)
    #obj = detector.run(**vars(opt))
   # perform inference with larger input size
#results = yolov5.predict(image1, size=1280, augment=True)
    # perform inference with test time augmentation
    results = yolov5.predict(img_uploaded_path,augment=True)
    print("detected obj:",results)
    column = results.pandas().xyxy[0].name
    print("records:",results.pandas().xyxy[0].to_json(orient="records"))
    # obj = {"person":0,"tie":0}
    jsonstr = column.value_counts().to_json()
    print("column:",column,"value_count:",column.value_counts(),jsonstr)
    jsonobj = json.loads(jsonstr)
    print("detected objson:",jsonobj)
        # obj = DeepFace.analyze(img_path = img_uploaded_path, actions = ['age', 'gender', 'race', 'emotion'],enforce_detection=False)
    end = datetime.now()
    exect = round(timeit.timeit(lambda:results),3)*1000
    print("The time of execution of above program is :",exect,"ms")
    jsonobj['execT'] = str(exect)+ "ms"
    print("detected response:",jsonobj)
    if not save_to_db:
        return jsonobj
    # then saving to mongodb
    # conn = pymongo.MongoClient('mongodb://root:example@mongo:27017/')

    savedata2db(jsonstr)
   

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
