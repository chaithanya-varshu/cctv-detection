# TensorFlow-Custom-Object-Detection : Training and Testing
<br>
Object detection allows for the recognition, detection, and localization of multiple objects within an image. It provides us a much better understanding of an image as a whole as apposed to just visual recognition.
<br>

<img src=validation/images/image1-detected.jpg height=350 width=50% /><img src=validation/images/test1-detected.jpg height=350 width=50% />
<br>
<h3>Requirements</h3>
<br>
You need to install all the below libraries in your python environment. You can do this in Python IDLE using simple pip command.
<h5>tensorflow, opencv, keras, imageai</h5>
<br>
<h3> 1. Label the images</h3>
Collect the images of different CC cameras, preferably in hundreds to create better model. You can use internet to find images or take pictures from your own devices.
<br>
Install the <b>labelimg</b> library and use the labelimg command to open the GUI application to label the images.
<br>

    >>> labelimg
    
<br>
We have to load each image one by one and mark the cc cams in the pictures and mark them as 'CC Cameras'. This cannot be automated. This will create xml files for each respective image wich same name as image name.
<h3> 2. Dataset Preparation</h3>
Have the folder structure as below. Refer to my repository for data. Both train and validation folders can be placed in a folder named dataset.
<br>

    >> train    >> images       >> img_1.jpg  (shows Object_1)
                >> images       >> img_2.jpg  (shows Object_2)
                >> images       >> img_3.jpg  (shows Object_1, Object_3 and Object_n)
                >> annotations  >> img_1.xml  (describes Object_1)
                >> annotations  >> img_2.xml  (describes Object_2)
                >> annotations  >> img_3.xml  (describes Object_1, Object_3 and Object_n)

    >> validation   >> images       >> img_151.jpg (shows Object_1, Object_3 and Object_n)
                    >> images       >> img_152.jpg (shows Object_2)
                    >> images       >> img_153.jpg (shows Object_1)
                    >> annotations  >> img_151.xml (describes Object_1, Object_3 and Object_n)
                    >> annotations  >> img_152.xml (describes Object_2)
                    >> annotations  >> img_153.xml (describes Object_1)

<br>
<h3> 3. Training the model</h3>
Use the following code to train the model, this code reads each image and its respective annotation and gets trained well.
<br>

    from imageai.Detection.Custom import DetectionModelTrainer
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="dataset")
    trainer.setTrainConfig(object_names_array=["CC Camera","Nest Cam"], batch_size=4, num_experiments=200,train_from_pretrained_model="pretrained-yolov3.h5")
    trainer.trainModel()

<br>
When i labeled the data using labelimg, i have used 2 classes 'CC Camera' and 'Nest Cam'. So here in object_names_array i have given same.
<br>
I used number of experiments as 200 so it trains the model 200 times. We can use a pretrained model, it is optional.
<br>
It takes much time based on your system speed.
<br>
After successful training a folder named 'models' will be created and you will see several models created in your dataset folder.
<br>
The output you will see something as below
<br>

    Generating anchor boxes for training images and annotation...
    Average IOU for 9 anchors: 0.80
    Anchor Boxes generated.
    Detection configuration saved in  dataset\json\detection_config.json
    Training on: 	['CC Camera', 'Nest Cam']
    Training with Batch Size:  4
    Number of Experiments:  200
    
    Epoch 1/5

     1/72 [..............................] - ETA: 1:43:08 - loss: 709.7822 - yolo_layer_1_loss: 94.0103 - yolo_layer_2_loss: 200.1460 - yolo_layer_3_loss: 415.6259
     2/72 [..............................] - ETA: 1:08:38 - loss: 708.1501 - yolo_layer_1_loss: 94.5144 - yolo_layer_2_loss: 197.7319 - yolo_layer_3_loss: 415.9039
     3/72 [>.............................] - ETA: 56:15

<br>

<h3> 4. Choosing the Best Model</h3>
We need to calculate the mAP value of each model based on these values we have to choose the best model to validate.
<br>Use the below code to find the mAP values.<br>

    from imageai.Detection.Custom import DetectionModelTrainer
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="dataset")
    metrics = trainer.evaluateModel(model_path="dataset/models", json_path="dataset/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
    print(metrics)

<br>

We will get output something like below
<br>

    Model File:  dataset/models\detection_model-ex-001--loss-0499.077.h5 

    Using IoU :  0.5
    Using Object Threshold :  0.3
    Using Non-Maximum Suppression :  0.5
    CC Camera: 0.0678
    Nest Cam: 0.0324
    mAP: 0.0774
    ===============================
    Model File:  dataset/models\detection_model-ex-002--loss-0150.152.h5 

    Using IoU :  0.5
    Using Object Threshold :  0.3
    Using Non-Maximum Suppression :  0.5
    CC Camera: 0.0124
    Nest Cam: 0.0330
    mAP: 0.0880

<br>
<h3> 5. Validating the Model</h3>
Now we will give an input image of a random cc camera, then our model will detect where are the cc cameras in that picture.
<br>To do that we have to provide the path where we are placing our image in the code.
<br>use the following code to validate the best model chosen above.
<br>

    from imageai.Detection.Custom import CustomObjectDetection
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("dataset/models/detection_model-ex-003--loss-0022.462.h5")
    detector.setJsonPath("dataset/json/detection_config.json")
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image="dataset/validation/images/test1.jpg", output_image_path="dataset/validation/images/test1-detected.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

<br>

You will see output something like
<br>

    Nest Cam  :  50.69343447685242  :  [75, 153, 255, 224]
    CC Camera  :  59.53064560890198  :  [232, 184, 286, 227]

<br>

<img src=validation/images/image1.jpg width=50% /><img src=validation/images/image1-detected.jpg width=50% />
<img src=validation/images/test1.jpg width=50% /><img src=validation/images/test1-detected.jpg width=50% />

