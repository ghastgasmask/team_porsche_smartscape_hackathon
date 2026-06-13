
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics YOLO no")

def main():
    if not YOLO_AVAILABLE:
        return

    # Check if roboflow dataset exists 
    # had to do this cuz it kept crashing
    dataset_yaml = "data/roboflow_dataset/data.yaml"
    if not os.path.exists(dataset_yaml):
        import glob
        matches = glob.glob("data/roboflow_dataset/**/data.yaml", recursive=True)
        if matches:
            dataset_yaml = matches[0]
            print(f"detected dataset config at {dataset_yaml}")
        else:
            print(f"Dataset not found at {dataset_yaml}")
            print("get a dataset")
            return

    print("Initializing YOLOv8 ")
    model = YOLO("yolov8n.pt")  

    print("Starting training ")
    model.train(
        data=dataset_yaml,
        epochs=20,          
        imgsz=640,
        project="models",
        name="yolo_urban"
    )
    print("Training complete and weights saved to models/yolo_urban/weights/best.pt (local)")

if __name__ == "__main__":
    main()
