from ultralytics.utils import LOGGER, SETTINGS, Path, is_ubuntu, get_ubuntu_version
from ultralytics.utils.checks import check_requirements, check_version
import fiftyone as fo
import fiftyone.zoo as foz
import warnings
from hydra import compose, initialize

def load_dataset(classes):

    check_requirements('fiftyone')
    
    if is_ubuntu() and check_version(get_ubuntu_version(), '>=22.04'):
    
    # Ubuntu>=22.04 patch https://github.com/voxel51/fiftyone/issues/2961#issuecomment-1666519347
        check_requirements('fiftyone-db-ubuntu2204')

    name = 'open-images-v7'
    fraction = 1.0  # fraction of full dataset to use
    for split in 'train', 'validation':  # 1743042 train, 41620 val images
        train = split == 'train'

        # Load Open Images dataset
        dataset = foz.load_zoo_dataset(name,
                                        split=split,
                                        label_types=['detections'],
                                        dataset_dir=Path(SETTINGS['datasets_dir']) / 'fiftyone' / name,
                                        max_samples=round((1743042 if train else 41620) * fraction),
                                        classes=classes)

        # Define classes
        if train:
            classes = dataset.distinct('ground_truth.detections.label')  # only observed classes

        # Export to YOLO format
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")
            dataset.export(export_dir=str(Path(SETTINGS['datasets_dir']) / name),
                            dataset_type=fo.types.YOLOv5Dataset,
                            label_field='ground_truth',
                            split='val' if split == 'validation' else split,
                            classes=classes,
                            overwrite=train)
            
if __name__ == '__main__':

    initialize(version_base=None, config_path="../../configs", job_name="app")
    cfg_dataset = compose(config_name="dataset")
    load_dataset(cfg_dataset.open_images_v7.classes)