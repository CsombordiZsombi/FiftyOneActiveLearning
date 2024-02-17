import fiftyone as fo
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO
import samplingstrategy as samp
import scoringfunction as scf
import dataset as ds
import yaml


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


def active_learning_loop(sampling_strategy, scoring_function, dataset, unlabeled_data, model,
                         export_name="exported_dataset", iterations=5, num_samples_to_query=25):
    """
    A method to automatically query the samples from the unlabeled dataset
    :param sampling_strategy: sampling strategies are available from the samplingstrategy.py
    :param scoring_function: scoring functions are available from the scoringfunction.py
    :param dataset: the train dataset on which the model will be trained
    :param unlabeled_data: a dataset, from which the unlabeled data will be queried
    :param model: the model used for predictions and training
    :param export_name: the name to export the train dateset to
    :param iterations: the number of iterations
    :param num_samples_to_query: the number of samples to query in one iteration
    :return:
    """

    # launch fiftyone app
    session = fo.launch_app(dataset, auto=False)  # TODO: remove
    print(session)

    # active learning loop
    for i in range(iterations):
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
    try:
        with open("FiftyOneActiveLearning/config.yaml", "r") as config_file:
            config = yaml.safe_load(config_file)

        num_start_samples = config["number of starting samples"]
        dataset = ds.load_dataset(dataset_name=config["dataset name"],
                                  dir_path=config["dataset path"],
                                  dataset_type=config["dataset type"],
                                  max_samples=config["number of starting samples"])

        if fo.dataset_exists(f"unlabeled_{config['dataset name']}"):
            fo.delete_dataset(f"unlabeled_{config['dataset name']}")

        unlabeled_data = ds.load_dataset(dataset_name=f"unlabeled_{config['dataset name']}",
                                         dir_path=config["dataset path"],
                                         dataset_type=config["dataset type"],
                                         max_samples=config["max samples"])

        for i in range(num_start_samples):
            unlabeled_data.delete_samples(unlabeled_data.first())  # deleting the redundant samples

        model = YOLO(config["model path"])

        active_learning_loop(export_name=f"export {config['dataset name']}",
                             sampling_strategy=samp.get_sampling_strategy_by_name(config["sampling strategy"]),
                             scoring_function=scf.get_scoring_function_by_name(config["scoring function"]),
                             dataset=dataset,
                             unlabeled_data=unlabeled_data,
                             model=model,
                             iterations=config["iterations"],
                             num_samples_to_query=config["number of samples to query"])

    except ValueError as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
    print("Done")
    input()
