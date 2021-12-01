import numpy as np
import logging


def intersect_lists(v1, v2, need_diff=True):
    """
        v1, v2: two lists of ints
        need_diff: if true, will return v2-v1 as well
        intersected: intersection of v1 and v2
        non_intersected: v2 - v1
    """
    intersected = []

    seen = set(v1)
    if need_diff:
        non_intersected = []
        for element in v2:
            if element in seen:
                intersected.append(element)
            else:
                non_intersected.append(element)
        return intersected, non_intersected
    else:
        for element in v2:
            if element in seen:
                intersected.append(element)
        return intersected


def summarize_scene(tracks, visible_tracks, visible_keypoints, coverage_thres,
                    alpha):
    logging.info("----------SummarizeScene Begins----------")

    assert len(visible_tracks) == len(visible_keypoints)
    num_images = len(visible_tracks)
    img_included = np.zeros((num_images,), dtype=np.bool)

    cur_coverage = 0

    covered_tracks = []
    confusing_tracks = []
    unique_tracks = []

    # greedily add an image to the iconic set
    # using coverage threshold as stopping criterion
    # TODO: try to use change in coverage as stopping criterion,
    # e.g. while delta >= ...
    while cur_coverage < coverage_thres:
        best_delta = -1e7
        chosen_image = -1
        chosen_intersected = None
        chosen_non_intersected = None

        for i in range(num_images):
            if img_included[i]:
                continue
            intersected, non_intersected = intersect_lists(
                covered_tracks, visible_tracks[i])

            cur_delta = len(non_intersected) - alpha * len(intersected)
            if (cur_delta > best_delta):
                best_delta = cur_delta
                chosen_image = i
                chosen_intersected = intersected
                chosen_non_intersected = non_intersected

        assert (chosen_image > -1)
        img_included[chosen_image] = True
        covered_tracks.extend(chosen_non_intersected)
        _, confusing_non_it = intersect_lists(confusing_tracks,
                                              chosen_intersected)
        confusing_tracks.extend(confusing_non_it)
        cur_coverage = len(covered_tracks) / len(tracks)

    _, unique_tracks = intersect_lists(confusing_tracks, covered_tracks)
    logging.info(f"{np.sum(img_included)} images included in the iconic set")
    logging.info(
        f"index: {[i for i, included in enumerate(img_included) if included]}")
    logging.info(f"there are {len(confusing_tracks)} confusing tracks "
                 f"and {len(unique_tracks)} unique tracks")
    logging.info("----------SummarizeScene Done----------")
    return unique_tracks, img_included
