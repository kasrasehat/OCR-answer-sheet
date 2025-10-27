from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

    model.train(data='config.yaml', epochs=70, imgsz=640, device='cuda:0')
    model.val()