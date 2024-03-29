from ultralytics import YOLO
import cv2
import torch
from datetime import date
import datetime

class Yolo8n():

    def __init__(self, conf, classes, path_model, path_vid):
        self.conf = conf
        self.classes = classes
        self.path_model = path_model
        self.path_vid = path_vid

    def start_work_str_vid(self):

        # Load the YOLOv8 model
        model = YOLO(self.path_model)

        # Open the video file
        video_path = self.path_vid
        results = model.predict(video_path, conf=self.conf, save_txt=True, save_conf=True, classes=self.classes, save=True,
                                        line_width=3)
        
        # cap = cv2.VideoCapture(video_path)

        # # Loop through the video frames
        # while cap.isOpened():
        #     # Read a frame from the video
        #     success, frame = cap.read()

        #     if success:
        #         # Run YOLOv8 inference on the frame
        #         results = model.predict(frame, conf=self.conf, save_txt=True, save_conf=True, classes=self.classes, save=True,
        #                                 line_width=3)

                # Visualize the results on the frame
                # annotated_frame = results[0].plot(conf=True)

                # # Display the annotated frame
                # cv2.imshow("YOLOv8 Inference", annotated_frame)

                # # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            # else:
            #     # Break the loop if the end of the video is reached
            #     break

        # Release the video capture object and close the display window
        # cap.release()
        # cv2.destroyAllWindows()

    def valid(self):

        model = YOLO(self.path_model)
        metrics = model.val()
        print(metrics.box.maps)

    def train(self):

        dataset_name = "/home/oem/Desktop/cv_detect_bags_experiments/data/dataset/coco.yaml"
        model = YOLO("yolov8n.pt")
        current_date = date.today()
        
        current_date_time = datetime.datetime.now()
        current_time = str(current_date_time.time()).split('.')[0]

        torch.cuda.empty_cache()

        model.train(
            data=dataset_name,
            epochs=300,
            patience=15,
            batch=16,
            imgsz=640,
            device='cuda',
            project=f'models/yolov8n_{current_date}_{current_time}'
        )