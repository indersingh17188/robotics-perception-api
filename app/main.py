from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import time
import uuid
import json
import base64
import html


app = FastAPI(
    title="Robotics Perception API",
    description="YOLO-based object detection API with JSON, browser view, and downloadable annotated image",
    version="0.2.0"
)

MODEL_VERSION = "yolov8n"
model = YOLO("yolov8n.pt")


def read_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def detect(image: Image.Image):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    results = model(image, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2)
                }
            })

    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "num_detections": len(detections),
        "inference_time_ms": inference_time_ms,
        "detections": detections
    }


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        bbox = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        text = f"{label} {conf:.2f}"

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=4
        )

        text_y = max(0, y1 - 20)

        draw.rectangle(
            [(x1, text_y), (x1 + len(text) * 9, text_y + 18)],
            fill="red"
        )

        draw.text(
            (x1 + 3, text_y + 2),
            text,
            fill="white"
        )

    return annotated


def image_to_jpeg_bytes(image: Image.Image) -> io.BytesIO:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


@app.get("/")
def home():
    return HTMLResponse("""
    <html>
        <head>
            <title>Robotics Perception API</title>
        </head>
        <body style="font-family: Arial; max-width: 900px; margin: auto;">
            <h2>Robotics Perception API</h2>
            <p>Upload an image to see YOLO detections, JSON output, and download annotated result.</p>

            <form action="/detect-view" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <button type="submit">Detect Objects</button>
            </form>

            <hr>
            <p>API docs: <a href="/docs">/docs</a></p>
            <p>Health check: <a href="/health">/health</a></p>
        </body>
    </html>
    """)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "task": "object_detection"
    }


@app.post("/detect-json")
async def detect_json(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image(image_bytes)
    result = detect(image)
    return result


@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image(image_bytes)
    result = detect(image)
    return result


@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    download: bool = Query(False)
):
    image_bytes = await file.read()
    image = read_image(image_bytes)

    result = detect(image)
    annotated = draw_boxes(image, result["detections"])

    output = image_to_jpeg_bytes(annotated)

    headers = {}
    if download:
        headers["Content-Disposition"] = "attachment; filename=detected_objects.jpg"
    else:
        headers["Content-Disposition"] = "inline; filename=detected_objects.jpg"

    return StreamingResponse(
        output,
        media_type="image/jpeg",
        headers=headers
    )


@app.post("/detect-view")
async def detect_view(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image(image_bytes)

    result = detect(image)
    annotated = draw_boxes(image, result["detections"])

    output = image_to_jpeg_bytes(annotated)
    encoded_image = base64.b64encode(output.getvalue()).decode("utf-8")

    json_text = html.escape(json.dumps(result, indent=2))

    return HTMLResponse(f"""
    <html>
        <head>
            <title>Detection Results</title>
        </head>
        <body style="font-family: Arial; max-width: 1000px; margin: auto;">
            <h2>Detection Results</h2>

            <p>
                <a href="/">Upload another image</a> |
                <a download="detected_objects.jpg" href="data:image/jpeg;base64,{encoded_image}">
                    Download annotated image
                </a>
            </p>

            <h3>Annotated Image</h3>
            <img src="data:image/jpeg;base64,{encoded_image}" style="max-width: 100%; border: 1px solid #ccc;">

            <h3>JSON Output</h3>
            <pre style="background:#f4f4f4; padding:15px; white-space:pre-wrap;">{json_text}</pre>
        </body>
    </html>
    """)