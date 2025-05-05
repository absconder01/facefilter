import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import os
import dlib

app = FastAPI()

# Check if model file exists
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model dosyası bulunamadı: {MODEL_PATH}. Lütfen https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat adresinden indirin.")

# Initialize detectors
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    predictor = dlib.shape_predictor(MODEL_PATH)
    detector = dlib.get_frontal_face_detector()
    
    if face_cascade.empty() or eye_cascade.empty():
        raise Exception("Cascade modelleri yüklenemedi")
except Exception as e:
    print(f"Model yükleme hatası: {str(e)}")
    face_cascade = None
    eye_cascade = None
    predictor = None
    detector = None

def create_skin_mask(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return mask

@app.post("/smooth")
async def smooth_image(image: UploadFile = File(...)):
    try:
        if face_cascade is None or eye_cascade is None or predictor is None:
            raise HTTPException(
                status_code=500,
                detail="Modeller yüklenemedi. Sistem hazır değil."
            )
            
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Geçersiz görüntü formatı")

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            # Yüz bulunamadıysa orijinal görüntüyü dön
            _, encoded_img = cv2.imencode('.jpg', img)
            return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

        # Create mask for face and skin
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        skin_mask = create_skin_mask(img)
        
        for (x, y, w, h) in faces:
            # Create face mask
            face_roi = mask[y:y+h, x:x+w]
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
            # Get face landmarks
            rect = dlib.rectangle(x, y, x+w, y+h)
            shape = predictor(gray, rect)
            
            # Detect and exclude eyes
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                eye_axes = (ew//2, eh//2)
                cv2.ellipse(mask, eye_center, eye_axes, 0, 0, 360, 0, -1)
            
            # Mask out lips using landmarks
            mouth_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            hull = cv2.convexHull(mouth_pts)
            cv2.fillConvexPoly(mask, hull, 0)
        
        # Combine face and skin masks
        final_mask = cv2.bitwise_and(mask, skin_mask)
        
        # Apply Gaussian blur to mask for smooth transitions
        final_mask = cv2.GaussianBlur(final_mask, (21, 21), 11)
        final_mask = final_mask.astype(float) / 255
        
        # Apply reduced smoothing
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        blur = cv2.GaussianBlur(smooth, (3, 3), 0)
        
        # Apply color enhancement
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.05)  # Reduced saturation enhancement
        s = np.clip(s, 0, 255)
        
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # Blend original and enhanced images using the mask
        final_mask = np.dstack([final_mask]*3)
        result = enhanced_img * final_mask + img * (1 - final_mask)
        result = result.astype(np.uint8)
        
        _, encoded_img = cv2.imencode('.jpg', result)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")
