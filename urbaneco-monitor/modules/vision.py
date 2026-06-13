import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not found")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics YOLO not found")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

RELEVANT_CLASSES = ["tree", "potted plant", "bench", "trash can", "car", "person",
                    "bicycle", "bus", "truck", "fire hydrant", "parking meter"]

CLASS_MAP = {
    "potted plant": "plant",
    "trash can": "trash",
    "fire hydrant": "infrastructure",
    "parking meter": "infrastructure",
}


def load_yolo_model(weights_path: str = None):
    if not YOLO_AVAILABLE:
        return None
    if weights_path and __import__("os").path.exists(weights_path):
        return YOLO(weights_path)
    return YOLO("yolov8n.pt")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required for image preprocessing")

    img = cv2.resize(image, (640, 640))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img.astype(np.float32) / 255.0


def detect_urban_objects(model, image: np.ndarray) -> dict:
    if model is None or not YOLO_AVAILABLE:
        return {}

    results = model(image, verbose=False)[0]
    detections: dict[str, list] = {}

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        simplified = CLASS_MAP.get(cls_name, cls_name)
        if cls_name not in RELEVANT_CLASSES and simplified not in RELEVANT_CLASSES:
            continue
        label = simplified
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        detections.setdefault(label, []).append({"bbox": bbox, "conf": round(conf, 3)})

    return detections


def calculate_green_ratio(image: np.ndarray, detections: dict = None) -> float:
    if not CV2_AVAILABLE:
        return 0.0

    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    ratio = np.sum(mask > 0) / mask.size * 100
    return round(float(ratio), 2)


def _draw_boxes(image: np.ndarray, detections: dict) -> np.ndarray:
    if not CV2_AVAILABLE:
        return image
    annotated = image.copy().astype(np.uint8)
    color_map = {
        "tree": (34, 139, 34),
        "plant": (50, 205, 50),
        "bench": (255, 165, 0),
        "trash": (220, 20, 60),
        "car": (70, 130, 180),
        "person": (148, 0, 211),
    }
    for label, boxes in detections.items():
        color = color_map.get(label, (200, 200, 200))
        for det in boxes:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{label} {det['conf']:.2f}",
                (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA,
            )
    return annotated


def analyze_image(image_input) -> dict:
    if isinstance(image_input, str):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required to load image from path")
        image = cv2.imread(image_input)
    elif PIL_AVAILABLE and isinstance(image_input, Image.Image):
        import numpy as _np
        image = _np.array(image_input.convert("RGB"))
        if CV2_AVAILABLE:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = image_input

    if image is None:
        return {"error": "Could not load image."}

    model = load_yolo_model()
    detections = detect_urban_objects(model, image)

    green_ratio = calculate_green_ratio(image, detections)

    trash_count = len(detections.get("trash", []))
    cleanliness_score = max(0.0, 100.0 - trash_count * 15.0)

    tree_count = len(detections.get("tree", [])) + len(detections.get("plant", []))
    vegetation_bonus = min(20.0, tree_count * 5.0)
    overall_vision_score = round(
        0.4 * cleanliness_score + 0.4 * green_ratio + 0.2 * vegetation_bonus, 1
    )
    overall_vision_score = min(100.0, overall_vision_score)

    annotated = _draw_boxes(image, detections)

    return {
        "detections": detections,
        "green_ratio": green_ratio,
        "cleanliness_score": round(cleanliness_score, 1),
        "overall_vision_score": overall_vision_score,
        "annotated_image": annotated,
        "object_counts": {k: len(v) for k, v in detections.items()},
    }
