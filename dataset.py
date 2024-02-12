import fiftyone as fo
import fiftyone.zoo as foz
import yaml


def yolov5_format_export(dataset, path_to):
    """
    Exports a dataset in yolov5 format.
    :param dataset: The dataset we want to export
    :param path_to: The path where we want to export to
    :return:
    """
    export_type = fo.types.YOLOv5Dataset
    test = dataset.load_saved_view("test")
    test.classes = dataset.classes
    test.export(
        export_dir=path_to,
        dataset_type=export_type,
        label_field=dataset.info["label_field"],
        split="test",
        classes=dataset.default_classes,
    )
    train = dataset.load_saved_view("train")
    train.classes = dataset.classes
    train.export(
        export_dir=path_to,
        dataset_type=export_type,
        label_field=dataset.info["label_field"],
        split="train",
        classes=dataset.default_classes,
    )
    val = dataset.load_saved_view("val")
    val.classes = dataset.classes
    val.export(
        export_dir=path_to,
        dataset_type=export_type,
        label_field=dataset.info["label_field"],
        split="val",  # Ultralytics uses 'val' others might use 'valid' or 'validation'
        classes=dataset.default_classes,
    )


def __init_splits(dataset, dir_path, label_field, default_classes, val_tag="val"):
    """
    Prepares the dataset to be easily exported
    Sets the default classes attribute to default_classes,
    the .info["path"] to dir_path,
    the .info["label_field"] to label_field
    and initializes the three splits

    :param dataset: The dataset we want to prepare for the active learning
    :param dir_path: The path of the directory containing the source files
    :param label_field: The name of the field identifying the ground truth detections
    :param default_classes: List of the classes names
    :param val_tag: The name of the tag identifying the validation split
    :return:
    """
    dataset.default_classes = default_classes
    dataset.info = {
        "path": dir_path,
        "label_field": label_field
    }
    dataset.save()
    if not dataset.has_saved_view("train"):
        dataset.save_view("train", dataset.match_tags("train", bool=True))
    if not dataset.has_saved_view("test"):
        dataset.save_view("test", dataset.match_tags("test", bool=True))
    if not dataset.has_saved_view("val"):
        dataset.save_view("val", dataset.match_tags(val_tag, bool=True))


def load_yolov5_format(dataset_name, dir_path, max_samples=None, **kwargs):
    """
    Loads a yolov5 type dataset.
    More info on the data structure:
    https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#yolov5dataset-import
    :param dataset_name: The name of the dataset
    :param dir_path: The path of the source files
    :param kwargs:
    :return: A fiftyone dataset
    """
    if fo.dataset_exists(dataset_name):
        print("Dataset already exist")
        return fo.load_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    splits = ["train", "test", "val"]

    for split in splits:
        dataset.add_dir(
            dataset_dir=dir_path,
            dataset_type=fo.types.YOLOv5Dataset,
            max_samples=max_samples,
            split=split,
            tags=split,
        )

    with open(dir_path + "/dataset.yaml", 'r') as stream:
        yaml_file = yaml.safe_load(stream)

    __init_splits(dataset, dir_path, default_classes=list(yaml_file["names"].values()), label_field="ground_truth")

    return dataset


def load_bbd100k(dataset_name, dir_path, max_samples=None):
    """
    Loads the bdd100k dataset from fiftyone-zoo.
    It will automatically download the dataset if it's not downloaded yet.

    :param dataset_name: The name of the dataset
    :param dir_path: The path of the directory containing the source files
    :param max_samples: The max amount of samples being loaded
    :return: A fiftyone dataset
    """
    if fo.dataset_exists(dataset_name):
        print("Dataset already exist, overwriting...")
        fo.delete_dataset(dataset_name)
        #return fo.load_dataset(dataset_name)

    dataset = foz.load_zoo_dataset(
        "bdd100k",
        source_dir=dir_path,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )
    default_classes = ["car", "traffic sign", "traffic light", "person", "truck", "bus", "rider", "motor", "bike",
                       "train"]
    __init_splits(dataset, dir_path, default_classes=default_classes, label_field="detections", val_tag="validation")

    return dataset


def load_coco_format(dataset_name, dir_path, max_samples=None, splits=None, val_tag=None):
    """
    Loads a Coco Detection type dataset.
    More info on the data structure: https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#cocodetectiondataset-import
    :param dataset_name: The name of the dataset
    :param dir_path: The path the directory containing the source files
    :param max_samples: The maximum number of samples to load
    :param splits: List of the names of the splits in the dataset
    :param val_tag: The name of the tag identifying the validation split

    :return: A fiftyone dataset
    """

    if fo.dataset_exists(dataset_name):
        print("Dataset already exist")
        return fo.load_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    val_tag = "valid" if not val_tag else None
    splits = ["train", "test", val_tag] if not splits else None

    for split in splits:
        dataset.add_dir(
            data_path=dir_path+"/"+split,
            dataset_type=fo.types.COCODetectionDataset,
            labels_path=dir_path+"/"+split+"/_annotations.coco.json",
            max_samples=max_samples,
            tags=split
        )

    __init_splits(dataset, dir_path, default_classes=dataset.default_classes, label_field="detections",
                  val_tag=val_tag)

    return dataset
