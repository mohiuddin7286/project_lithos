from ultralytics import YOLO

print("--- PROJECT LITHOS: Drone Vision Training Initiated ---\n")

# 1. Initialize a blank, cutting-edge YOLOv8 Neural Network
print("Loading YOLOv8 Nano Architecture...")
model = YOLO('yolov8n.pt') 

# 2. Train the AI on your images
# (You will need a file named 'dataset.yaml' that points to your images folder)
print("Commencing Deep Learning sequence. This will take time!")
results = model.train(
    data='dataset.yaml',   # Config file telling YOLO where your images are
    epochs=10,             # Shorter CPU-friendly run for faster completion
    imgsz=416,             # Smaller image size reduces training time on CPU
    batch=8,               # Smaller batches fit CPU memory more comfortably
    fraction=0.25,         # Use a subset for a practical first training pass
    workers=0,             # More reliable on Windows when running from VS Code
    patience=10,
    device='cpu'           # Change to '0' if you have an Nvidia GPU!
)

print("\n💾 SUCCESS: Vision Model trained and saved to 'runs/detect/train/weights/best.pt'")