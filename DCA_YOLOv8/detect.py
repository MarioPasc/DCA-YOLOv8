from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    model = YOLO.classify('yolov8n.yaml')

    # Train the model
    result = model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)