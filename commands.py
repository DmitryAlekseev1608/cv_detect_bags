from hydra import compose, initialize
import fire
import torch

from cv_detect_bags_experiments.yolo8n import Yolo8n
from cv_detect_bags_experiments.dataset_coco import DatasetCoco

def start_yolov8n_pret():
    yolov8n_pret =  Yolo8n(cfg_yolo8n.yolov8n_pret.conf, cfg_yolo8n.yolov8n_pret.classes, cfg_yolo8n.yolov8n_pret.path_model, \
                           cfg_yolo8n.data.path_vid)
    yolov8n_pret.start_work_str_vid()
   
def start_yolov8n_alex():
    yolo8n_alex = Yolo8n(cfg_yolo8n.yolo8n_alex.conf, cfg_yolo8n.yolo8n_alex.classes, cfg_yolo8n.yolo8n_alex.path_model, \
                         cfg_yolo8n.data.path_vid)
    yolo8n_alex.start_work_str_vid()

def start_cv_detect_bags_yolo8n_one_class_coco2017():
    cv_detect_bags_yolo8n_one_class_coco2017 = Yolo8n(cfg_yolo8n.cv_detect_bags_yolo8n_one_class_coco2017.conf,
                                                      cfg_yolo8n.cv_detect_bags_yolo8n_one_class_coco2017.classes,
                                                      cfg_yolo8n.cv_detect_bags_yolo8n_one_class_coco2017.path_model, \
                                                      cfg_yolo8n.data.path_vid)
    cv_detect_bags_yolo8n_one_class_coco2017.start_work_str_vid()

def start_load_dataset_coco():
    dataset_coco = DatasetCoco(cfg_dataset_coco.param.segments, cfg_dataset_coco.param.dataset_root_dir)
    dataset_coco.load_dataset_coco()

def start_watch_param_model(model):
    model = torch.load(f"models/{model}")
    print(model)

if __name__ == '__main__':

    initialize(version_base=None, config_path="configs", job_name="app")
    cfg_yolo8n = compose(config_name="yolo8n")
    cfg_dataset_coco = compose(config_name="dataset_coco")
    fire.Fire()
