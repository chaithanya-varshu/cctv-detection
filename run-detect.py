
import warnings
warnings.filterwarnings("ignore")
from imageai.Detection.Custom import CustomObjectDetection
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("dataset/models/detection_model-ex-003--loss-0022.462.h5")
detector.setJsonPath("dataset/json/detection_config.json")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="dataset/validation/images/test1.jpg", output_image_path="dataset/validation/images/test1-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
