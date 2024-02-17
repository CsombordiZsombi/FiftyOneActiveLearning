from operator import attrgetter
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from sklearn.cluster import KMeans


def min_samples(
                pool,
                scoring_function,
                n_samples,
                score_tag="score",
                prediction_tag="predictions",
                progression_bar=True,
                **kwargs
                ):
    """
    Chooses n samples with the lowest scores.
    :param pool: A pool of samples
    :param scoring_function: The function scoring the samples
    :param n_samples: The number of samples to query
    :param score_tag: Which tag to use to store the scores
    :param prediction_tag: Which tag to use to reach the predictions
    :param progression_bar: Enable progression bar
    :return: An array of the chosen samples
    """

    data_points_to_query = []

    # make predictions on the pool
    scoring_function(pool, score_tag=score_tag, prediction_tag=prediction_tag, progress_bar=progression_bar)
    if progression_bar:
        print("Sampling")
    for sample in pool.iter_samples(progress=progression_bar):
        if sample[score_tag] == 0:
            continue  # TODO do something when there are no detection
        # fill up the array
        if len(data_points_to_query) < n_samples:
            data_points_to_query.append(sample)
        else:

            # pick the best samples
            max_sample = max(data_points_to_query, key=attrgetter(score_tag))
            if max_sample[score_tag] > sample[score_tag]:
                data_points_to_query.remove(max_sample)
                data_points_to_query.append(sample)

    return data_points_to_query


def max_samples(
                pool,
                scoring_function,
                n_samples,
                score_tag="score",
                prediction_tag="predictions",
                progression_bar=True,
                **kwargs
                ):
    """
    Chooses n samples with the highest scores.
    :param pool: A pool of samples
    :param scoring_function: The function scoring the samples
    :param n_samples: The number of samples to query
    :param score_tag: Which tag to use to store the scores
    :param prediction_tag: Which tag to use to reach the predictions
    :param progression_bar: Enable progression bar
    :return: An array of the chosen samples
    """

    data_points_to_query = []

    # predicting the pool's content
    scoring_function(pool, score_tag=score_tag, prediction_tag=prediction_tag, progress_bar=progression_bar)
    if progression_bar:
        print("Sampling")
    for sample in pool.iter_samples(progress=progression_bar):
        if sample[score_tag] == 0:  # TODO do something when there are no detections
            continue
        # filling up the array
        if len(data_points_to_query) < n_samples:
            data_points_to_query.append(sample)
        else:
            # choosing the best samples
            min_sample = min(data_points_to_query, key=attrgetter(score_tag))
            if min_sample[score_tag] < sample[score_tag]:
                data_points_to_query.remove(min_sample)
                data_points_to_query.append(sample)

    return data_points_to_query


def k_means(
            pool,
            scoring_function,
            n_samples,
            sampling_strategy=min_samples,
            score_tag="score",
            prediction_tag="predictions",
            progression_bar=True,
            num_clusters=10,
            **kwargs
            ):
    """
    Uses k-means make clusters, then chooses evenly distributed samples from the clusters
    :param pool: A pool of samples
    :param scoring_function: The function scoring the samples
    :param n_samples: The number of samples to query
    :param sampling_strategy: The strategy to use to query from the clusters
    :param score_tag: Which tag to use to store the scores
    :param prediction_tag: Which tag to use to reach the predictions
    :param progression_bar: Enable progression bar
    :param num_clusters: The number of clusters to use
    :return: An array of the chosen samples
    """

    # creating the embeddings -> simplifying the samples

    model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    embeddings = pool.compute_embeddings(model)

    # creating the clusters
    clusters = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

    # storing the clusters
    pool.set_values("cluster_label", clusters)

    samples_to_query = []

    # we iterate through the clusters
    for cluster in range(num_clusters):
        # filter for the cluster
        cluster_view = pool.match(F("cluster_label") == cluster)

        # pick the best samples from the cluster
        query_from_cluster = sampling_strategy(pool=cluster_view,scoring_function=scoring_function,
                                               n_samples=n_samples/num_clusters, score_tag=score_tag,
                                               prediction_tag=prediction_tag,progression_bar=progression_bar)

        samples_to_query.extend(query_from_cluster)

    return samples_to_query


def get_sampling_strategy_by_name(name):
    """
    Returns a sampling strategy by name. If no match found, raises a ValueError
    :param name: The name of the sampling strategy
    :return: a function
    """
    match name:
        case "k_means":
            return k_means
        case "max_samples":
            return max_samples
        case "min_samples":
            return min_samples
        case _:
            raise ValueError(f"No such sampling strategy: {name}")

