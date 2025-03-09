from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io

app = FastAPI()

# Загружаем модель YOLO (можно заменить на другую версию, например 'yolov8n.pt')
model = YOLO("yolov8n.pt")

# Список классов, относящихся к животным
ANIMAL_CLASSES = {"cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

@app.post("/detect")
async def detect_animals(file: UploadFile = File(...)):
    try:
        # Читаем изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Запускаем детекцию объектов
        results = model(image)
        draw = ImageDraw.Draw(image)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name in ANIMAL_CLASSES:
                    bbox = box.xyxy[0].tolist()
                    draw.rectangle(bbox, outline="red", width=3)
                    draw.text((bbox[0], bbox[1]), class_name, fill="red")
        
        # Сохраняем изображение в буфер
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        return Response(content=img_bytes.getvalue(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
