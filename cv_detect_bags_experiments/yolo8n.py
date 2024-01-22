from ultralytics import YOLO
import cv2
import torch

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
        cap = cv2.VideoCapture(video_path)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame
                results = model.predict(frame, conf=self.conf, save_txt=True, save_conf=True, classes=self.classes)

                # Visualize the results on the frame
                annotated_frame = results[0].plot(conf=True)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def valid(self):

        model = YOLO(self.path_model)
        metrics = model.val()
        print(metrics.box.maps)

    def train(self):

        dataset_name = "data/dataset/coco.yaml"
        model = YOLO("yolov8n.pt")

        torch.cuda.empty_cache()

        model.train(
            data=dataset_name,
            epochs=300,
            patience=15,
            batch=16,
            imgsz=640,
            device='cuda',
            project='models',
            name = 'Dataset: coco2017, roboflow bags, open images v7; Class: bag'
        )