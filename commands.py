from hydra import compose, initialize
import fire
import torch

from cv_detect_bags_experiments.yolo8n import Yolo8n
from cv_detect_bags_experiments.dataset_coco import DatasetCoco

def cv_detect_bags_yolo8n():
    cv_detect_bags_yolo8n = Yolo8n(cfg_yolo8n.yolo8n.conf, cfg_yolo8n.yolo8n.classes, cfg_yolo8n.yolo8n.path_model, \
                                                      cfg_yolo8n.data.path_vid)
    cv_detect_bags_yolo8n.start_work_str_vid()

def load_dataset_coco():
    dataset_coco = DatasetCoco(cfg_dataset_coco.param.segments, cfg_dataset_coco.param.dataset_root_dir)
    dataset_coco.load_dataset_coco()

def watch_param_model(model):
    model = torch.load(f"models/{model}")
    print(model)

if __name__ == '__main__':

    initialize(version_base=None, config_path="configs", job_name="app")
    cfg_yolo8n = compose(config_name="yolo8n")
    cfg_dataset_coco = compose(config_name="dataset_coco")
    fire.Fire()
