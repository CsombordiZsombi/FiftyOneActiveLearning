import fiftyone.zoo as foz
import fiftyone as fo
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO
import samplingstrategy as samp
import scoringfunction as scf
import dataset as ds


def predict(dataset, model, prediction_tag="predictions", iou=0.5):
    """
    Makes prediction on the dataset and saves them in the sample's prediction_tag
    :param dataset: A fiftyone dataset or dataset_view
    :param model: An ultralitics model used for the predicitons
    :param prediction_tag: The tag were we save the predictions
    :param iou: Intersection over union for less bounding boxes
    :return:
    """

    for sample in dataset.iter_samples(progress=True):
        result = model.predict(sample.filepath, iou=iou, verbose=False, device=0)[0]
        sample[prediction_tag] = fou.to_detections(result)
        sample.save()


def query(database, samples_to_query, unlabeled_data=None):
    """
    A method to automatically query the samples from the database
    :param database: A fiftyone database to get the labeled samples from
    :param samples_to_query: An array of samples
    :param unlabeled_data: If provided, the queried samples will be removed from it
    :return:
    """
    if unlabeled_data is not None:
        unlabeled_data.delete_samples(samples_to_query)
    database.add_samples(samples_to_query)


def active_learning_loop(sampling_strategy, scoring_function, export_name,
                         iterations=20, num_samples_to_query=250):
    num_start_samples = 1000
    dataset = ds.load_bbd100k("bdd_set", "C:/Users/Zsombor/fiftyone/bdd100k", max_samples=num_start_samples)
    if fo.dataset_exists("unlabeled_data"):
        fo.delete_dataset("unlabeled_data")
    unlabeled_data = foz.load_zoo_dataset(
        "bdd100k",
        source_dir="C:/Users/Zsombor/fiftyone/bdd100k",
        max_samples=10000,
        dataset_name="unlabeled_data",
        split="train",
    )
    for i in range(num_start_samples):
        unlabeled_data.delete_samples(unlabeled_data.first()) # deleting the

    model = YOLO("yolov8s.pt")

    # launch fiftyone app
    session = fo.launch_app(dataset, auto=False)  # TODO: remove
    print(session)

    # active learning loop
    for i in range(iterations):
        export_dir = str(i)
        # making predictions on all the samples in the unlabeled dataset
        print("predicting:")
        predict(unlabeled_data, model)

        # choosing the best samples and querying them
        print("querying:")
        samples_to_query = sampling_strategy(pool=unlabeled_data, scoring_function=scoring_function,
                                             n_samples=num_samples_to_query)
        query(dataset, samples_to_query, unlabeled_data)

        # exporting the dataset to be able to teach the model
        print("exporting sets:")
        ds.yolov5_format_export(dataset, f'yolo_dataset/{export_name}/')

        # training the model
        print("training model:")
        model.train(data=f'yolo_dataset/{export_name}/dataset.yaml', epochs=3, save=False, verbose=False, batch=4)

        print(str(i+1) + " active learning iteration ended")


def main():

    active_learning_loop(export_name="min_samples-average_confidence",
                         sampling_strategy=samp.min_samples,
                         scoring_function=scf.average_confidence,
                         iterations=10,
                         num_samples_to_query=500)


if __name__ == "__main__":
    main()
    print("Done")
    input()

