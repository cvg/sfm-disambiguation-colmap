import numpy as np
import logging
import time
from multiprocessing import Pool
from functools import partial

from .utils.database import COLMAPDatabase
from .utils.read_write_database import (read_image_names_from_db,
                                        read_intrinsics_from_db)
from .missing_correspondences import (reconstruct_depths, estimate_scales,
                                      check_depth_consistency,
                                      calculate_missing_scores_v1,
                                      calculate_missing_scores_v2,
                                      calculate_missing_scores_v3)


def main(old_db_path,
         use_E=False,
         min_num_valid_depths=5,
         max_num_neighbors=80,
         square_radius=20,
         score_version=-1,
         parallel=True,
         image_path=None,
         plot_path=None):
    old_db = COLMAPDatabase.connect(old_db_path)
    id0s = [id0 for id0, in old_db.execute("SELECT image_id FROM images")]
    num_images1 = len(id0s)
    num_images = next(old_db.execute("SELECT COUNT(*) FROM images"))[0]
    assert num_images1 == num_images
    image_names = read_image_names_from_db(old_db)
    intrinsics = read_intrinsics_from_db(old_db)
    old_db.close()
    if score_version == 1:
        calculate_missing_scores_f = calculate_missing_scores_v1
    elif score_version == 2:
        calculate_missing_scores_f = calculate_missing_scores_v2
    elif score_version == 3:
        calculate_missing_scores_f = calculate_missing_scores_v3
    else:
        raise NotImplementedError
    partial_calculate_score_for_one_image = partial(
        calculate_score_for_one_image,
        calculate_missing_scores_f=calculate_missing_scores_f,
        num_images=num_images,
        old_db_path=old_db_path,
        intrinsics=intrinsics,
        use_E=use_E,
        max_num_neighbors=max_num_neighbors,
        square_radius=square_radius,
        image_path=image_path,
        image_names=image_names,
        plot_path=plot_path)

    num_valid_depths = np.zeros((num_images, num_images), dtype=np.int)
    scores = np.zeros((num_images, num_images), dtype=np.float64)
    reconstruct_depths_time = 0
    estimate_scales_time = 0
    check_depth_consistency_time = 0
    calculate_missing_scores_time = 0
    if parallel:
        with Pool() as pool:
            return_values = pool.map(partial_calculate_score_for_one_image,
                                     id0s)
            for return_value in return_values:
                (id0, scores_id0, num_valid_depths_id0, t01, t12, t23,
                 t34) = return_value
                scores[id0 - 1] = scores_id0
                num_valid_depths[id0 - 1] = num_valid_depths_id0
                reconstruct_depths_time += t01
                estimate_scales_time += t12
                check_depth_consistency_time += t23
                calculate_missing_scores_time += t34
    else:
        for id0 in id0s:
            (_, scores[id0 - 1], num_valid_depths[id0 - 1], t01, t12, t23,
             t34) = partial_calculate_score_for_one_image(id0)
            reconstruct_depths_time += t01
            estimate_scales_time += t12
            check_depth_consistency_time += t23
            calculate_missing_scores_time += t34

    scores[num_valid_depths < min_num_valid_depths] = 0.
    suffix = (f'_cui_v{score_version}_d{min_num_valid_depths}' +
              f'_n{max_num_neighbors}_r{square_radius}')
    if parallel:
        suffix += '_p'
    np.save(old_db_path.parent / ('scores' + suffix + '.npy'), scores)

    logging.info("----------------Time Analysis---------------------")
    logging.info(
        f'Reconstuct Depths: {reconstruct_depths_time / 60:.2f} minutes')
    logging.info(f'Estimate Scales: {estimate_scales_time / 60:.2f} minutes')
    logging.info('Check Depth Consistency: '
                 f'{check_depth_consistency_time / 60:.2f} minutes')
    logging.info('Calculate Missing Scores: '
                 f'{calculate_missing_scores_time / 60:.2f} minutes')
    return


def calculate_score_for_one_image(id0,
                                  calculate_missing_scores_f,
                                  num_images,
                                  old_db_path,
                                  intrinsics,
                                  use_E,
                                  max_num_neighbors,
                                  square_radius=0,
                                  image_path=None,
                                  image_names=None,
                                  plot_path=None):
    t0 = time.time()
    old_db = COLMAPDatabase.connect(old_db_path)
    (kpts0_c, depths_list, neighbor_ids, kpts0_matches,
     poses) = reconstruct_depths(old_db, id0, intrinsics, max_num_neighbors,
                                 use_E)
    t1 = time.time()

    scales, invalid_neighbor_ids = estimate_scales(depths_list, neighbor_ids)
    if scales is None:
        t2 = time.time()
        t3 = time.time()
        t4 = time.time()
        scores = np.zeros((num_images,))
        num_valid_depths = np.zeros((num_images,), dtype=np.int)
        return id0, scores, num_valid_depths, t1 - t0, t2 - t1, t3 - t2, t4 - t3
    for invalid_neighbor_id in invalid_neighbor_ids:
        poses.pop(invalid_neighbor_id)
        kpts0_matches.pop(invalid_neighbor_id)
    # remove reconstructed points by invalid neighbors
    # by definition this point will only have one reconstruction
    for i in range(len(depths_list)):
        if (len(depths_list[i]) > 0 and
                depths_list[i][0][0] in invalid_neighbor_ids):
            assert len(depths_list[i]) == 1
            depths_list[i] = []
    t2 = time.time()

    median_depths, num_valid_depths = check_depth_consistency(
        num_images, scales, depths_list)
    t3 = time.time()

    scores = calculate_missing_scores_f(num_images=num_images,
                                        id0=id0,
                                        median_depths=median_depths,
                                        kpts0_c=kpts0_c,
                                        scales=scales,
                                        poses=poses,
                                        kpts0_matches=kpts0_matches,
                                        intrinsics=intrinsics,
                                        square_radius=square_radius,
                                        image_path=image_path,
                                        image_names=image_names,
                                        plot_path=plot_path)
    old_db.close()
    t4 = time.time()

    return id0, scores, num_valid_depths, t1 - t0, t2 - t1, t3 - t2, t4 - t3
