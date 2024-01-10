from hydra import compose, initialize
import fire

from cv_detect_bags_experiments.Yolo8n import Yolo8n

def start_yolov8n_pret():
    yolov8n_pret =  Yolo8n(cfg.yolov8n_pret.conf, cfg.yolov8n_pret.classes, cfg.yolov8n_pret.path_model)
    yolov8n_pret.start_work_str_vid()
   
def start_yolov8n_alex():
    yolo8n_alex = Yolo8n(cfg.yolo8n_alex.conf, cfg.yolo8n_alex.classes, cfg.yolo8n_alex.path_model)
    yolo8n_alex.start_work_str_vid()

if __name__ == '__main__':

    initialize(version_base=None, config_path="configs", job_name="app")
    cfg = compose(config_name="config")
    fire.Fire()
