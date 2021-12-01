import time
import logging
import numpy as np

from .utils.database import COLMAPDatabase
from .geodesic_consistency import MatchesList
from .geodesic_consistency import (compute_tracks, construct_path_network,
                                   summarize_scene)


def main(old_db_path,
         track_degree=3,
         coverage_thres=0.6,
         alpha=0.,
         minimal_views=5,
         ds='largearray'):
    """
    Compute the score between images based on paper "Distinguishing the
    Indistinguishable: Exploring Structural Ambiguities via Geodesic Context"
    by Yan et al.

    Args:
        old_db_path: path to the original colmap database
        track_degree: minimal #views in different images
                      for a track to be valid
        coverage_thres: minimal percentage for the tracks
                            covered by the iconic set
        alpha: the weight for distinctiveness term in the objective function
               of choosing the next image to be added to the iconic set
        minimal_views: minimal #unique tracks two images must share
                       to be regarded as geodesically consistent
        ds: data structure to store matches_list.
            'dict' is fast but memory intensive.
            'smallarray' requires much less memory but is quite slow.
            'largearray' is a tradeoff between two ds mentioned above
    Saved:
        scores: 0-based n x n matrices where scores[i][j] is the match score
        between image i+1 and image j+1
    """
    old_db = COLMAPDatabase.connect(old_db_path)
    num_images = next(old_db.execute("SELECT COUNT(*) FROM images"))[0]
    keypoints_rows = old_db.execute("SELECT rows FROM keypoints")
    num_keypoints_list = [row[0] for row in keypoints_rows]
    max_num_keypoints = max(num_keypoints_list)
    old_db.close()

    t0 = time.time()
    matches_list = MatchesList(num_images, max_num_keypoints, ds, old_db_path)
    t1 = time.time()

    tracks, visible_tracks, visible_keypoints = compute_tracks(
        num_images, num_keypoints_list, matches_list, track_degree, ds)
    t2 = time.time()

    unique_tracks, img_included = summarize_scene(tracks, visible_tracks,
                                                  visible_keypoints,
                                                  coverage_thres, alpha)
    t3 = time.time()

    scores = construct_path_network(num_images, matches_list, img_included,
                                    unique_tracks, visible_tracks,
                                    minimal_views)
    t4 = time.time()
    suffix = (
        f'_yan_t{track_degree}_c{coverage_thres}_a{alpha}_m{minimal_views}')
    np.save(old_db_path.parent / ('scores' + suffix + '.npy'), scores)

    logging.info("----------------Time Analysis---------------------")
    logging.info(f"Sort Matches: {(t1 - t0) / 60:.2f} minutes")
    logging.info(f"Compute Tracks: {(t2 - t1) / 60:.2f} minutes")
    logging.info(f"Summarize Scene: {(t3 - t2) / 60:.2f} minutes")
    logging.info(f"Construct Path Network: {(t4 - t3) / 60:.2f} minutes")
