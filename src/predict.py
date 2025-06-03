from ultralytics import YOLO

if __name__ == "__main__":
    CKPT_PATH = "models/best.pt" # 假设best.pt在此路径
    # CKPT_PATH = "runs/detect/train4/weights/best.pt" # 训练后生成的best.pt路径

    IMGSZ = (256, 256) # 预测时使用的图片大小

    # 加载模型
    model = YOLO(CKPT_PATH)

    # 对 'img/test' 目录下的所有图片进行预测
    # save=True 会将预测结果（带有边界框的图片）保存到 runs/detect/predict 目录下
    # conf=0.25 是一个默认的置信度阈值，您可以根据需要调整
    model.predict('img/test', save=True, imgsz=IMGSZ, conf=0.25)