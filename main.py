import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

app = FastAPI()

@app.post("/smooth")
async def smooth_image(image: UploadFile = File(...)):
    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    smooth = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)

    blur = cv2.GaussianBlur(smooth, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.1)
    s = np.clip(s, 0, 255)

    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    blended = cv2.addWeighted(img, 0.5, enhanced_img, 0.5, 0)

    _, encoded_img = cv2.imencode('.jpg', blended)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
