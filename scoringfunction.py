import fiftyone.brain as fob


def average_confidence(pool,
                       progress_bar=True,
                       score_tag="score",
                       prediction_tag="predictions",
                       **kwargs
                       ):
    """
    Takes the average confidence of the predicted bounding boxes and stores them in the score tag of the samples.
    :param pool: The samples containing the predictions
    :param progress_bar: Enable progress bar
    :param score_tag: Where to store the scores to a sample
    :param prediction_tag: The tag containing the bounding boxes
    :param kwargs:
    :return:
    """
    if progress_bar:
        print("Computing scores")
    for sample in pool.iter_samples(progress=progress_bar):
        score_sum = 0
        for detection in sample[prediction_tag].detections:
            score_sum += detection.confidence
        num_of_detections = len(sample[prediction_tag].detections)
        avg_conf = 0
        if num_of_detections != 0:
            avg_conf = score_sum / num_of_detections
        sample[score_tag] = avg_conf
        sample.save()


def least_confident(pool,
                    progress_bar=True,
                    score_tag="score",
                    prediction_tag="predictions",
                    **kwargs
                    ):
    """
    Takes the bounding box with the lowest confidence and use the confidence as the score.
    Stores the score in the score tag of the samples.
    :param pool: The samples containing the predictions
    :param progress_bar: Enable progress bar
    :param score_tag: Where to store the scores to a sample
    :param prediction_tag: The tag containing the bounding boxes
    :param kwargs:
    :return:
    """
    if progress_bar:
        print("Computing scores")
    for sample in pool.iter_samples(progress=progress_bar):
        min_conf = 1
        for detection in sample[prediction_tag].detections:
            min_conf = min(min_conf, detection.confidence)
        sample[score_tag] = min_conf
        sample.save()


def most_confident(pool,
                   progress_bar=True,
                   score_tag="score",
                   prediction_tag="predictions",
                   **kwargs
                   ):
    """
    Takes the bounding box with the highest confidence and use the confidence as the score.
    Stores the score in the score tag of the samples.
    :param pool: The samples containing the predictions
    :param progress_bar: Enable progress bar
    :param score_tag: Where to store the scores to a sample
    :param prediction_tag: The tag containing the bounding boxes
    :param kwargs:
    :return:
    """
    if progress_bar:
        print("Computing scores")
    for sample in pool.iter_samples(progress=progress_bar):
        max_conf = 0
        for detection in sample[prediction_tag].detections:
            max_conf = max(max_conf, detection.confidence)
        sample[score_tag] = max_conf
        sample.save()


def uniqueness_score(pool,
                     score_tag="score",
                     **kwargs
                     ):
    """
    Calculates a uniqueness score using fiftyone.brain and stores them in the score tag of the samples.
    :param pool: The samples containing the predictions
    :param score_tag: Where to store the scores to a sample
    :param kwargs:
    :return:
    """
    fob.compute_uniqueness(pool, uniqueness_field=score_tag)


def get_scoring_function_by_name(name):
    """
    Returns the matching scoring function for the given name, or raise ValueError if no matching found
    :param name: The name of the scoring function
    :return: a function
    """
    match name:
        case "uniqueness_score":
            return uniqueness_score
        case "most_confident":
            return most_confident
        case "average_confidence":
            return average_confidence
        case "least_confident":
            return least_confident
        case _:
            raise ValueError(f"No scoring function: {name}")
