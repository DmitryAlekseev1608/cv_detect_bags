from ultralytics.utils.downloads import download
from pathlib import Path

class DatasetCoco():

    ''' Ultralytics YOLO 🚀, AGPL-3.0 license
    COCO 2017 dataset http://cocodataset.org by Microsoft
    Documentation: https://docs.ultralytics.com/datasets/detect/coco/
    Example usage: yolo train data=coco.yaml
    parent
    ├── ultralytics
    └── datasets
        └── coco  ← downloads here (20.1 GB)
    Download script/URL (optional)
    Download labels'''

    def __init__(self, segments, dataset_root_dir):
        self.segments = segments
        self.dataset_root_dir = dataset_root_dir
    
    def load_dataset(self):
   
        segments = self.segments  # segment or box labels
        dir = Path(self.dataset_root_dir)  # dataset root dir
        url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
        urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
        download(urls, dir=dir.parent)
        
        # Download data
        urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
                'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
                'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
        download(urls, dir=dir / 'images', threads=3)