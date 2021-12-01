import numpy as np
import logging

from .summarize_scene import intersect_lists


def construct_path_network(num_images, matches_list, img_included,
                           unique_tracks, visible_tracks, minimal_views):
    logging.info("----------ConstructPathNetwork Starts----------")
    scores = np.zeros((num_images, num_images))
    for i in range(num_images):
        if img_included[i]:
            logging.info("---iconic image {} connects:".format(i + 1))
        else:
            logging.info("---non-iconic image {} connects:".format(i + 1))
        unique_i = intersect_lists(unique_tracks,
                                   visible_tracks[i],
                                   need_diff=False)
        for j in matches_list.get_neighbors(i):
            # skip non-iconic images, was commented out in the original code
            # if we use continue here, then there will be matches between
            # non-iconic images
            # if not img_included[j-1]:
            #     continue

            # i is 0-based and j is 1-based
            unique_j = intersect_lists(unique_tracks,
                                       visible_tracks[j - 1],
                                       need_diff=False)
            unique_ij = intersect_lists(unique_i, unique_j, need_diff=False)
            if len(unique_ij) > minimal_views:
                score = len(unique_ij) / max(len(unique_i), len(unique_j))
                scores[i, j - 1] = score
                scores[j - 1, i] = score
                logging.info(
                    f"image {j} (common unique points {len(unique_ij)}, "
                    f"max unique points {max(len(unique_i),len(unique_j))}"
                    f", score {score:.2f})")

    logging.info("----------ConstructPathNetwork Done----------")
    return scores
