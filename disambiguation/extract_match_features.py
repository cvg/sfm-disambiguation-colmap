from disambiguation import set_logger
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.reconstruction import import_images

from .utils.database import COLMAPDatabase
from .utils.run_colmap import (run_feature_extractor, run_exhaustive_matcher,
                               run_matches_importer)
from .utils.read_write_database import (write_keypoints_into_db,
                                        write_matches_into_db)
from .options.feature_options import hloc_feature_confs
from .options.matching_options import hloc_matching_confs


def main(feature_type, matching_type, geometric_verification_type,
         dataset_path, results_path, colmap_path, db_path, use_gpu, log_path):
    assert feature_type in [
        'sift_default', 'sift_strict', 'superpoint', 'r2d2', 'd2net', 'disk'
    ]
    assert matching_type in ['sift_default', 'sift_strict', 'nn', 'superglue']
    assert geometric_verification_type in ['strict', 'default']
    image_path = dataset_path / 'images'
    results_path.mkdir(parents=True, exist_ok=True)

    set_logger(log_path)

    # extract features via colmap or hloc
    if 'sift' in feature_type:
        colmap_sift_type = feature_type.split('_')[1]
        run_feature_extractor(colmap_path, db_path, image_path,
                              colmap_sift_type, use_gpu)
    else:
        feature_conf = hloc_feature_confs[feature_type]
        feature_path = extract_features.main(feature_conf, image_path,
                                             results_path)

    # match features via colmap or hloc
    # TODO: add sequential matching?
    if 'sift' in matching_type:
        colmap_matching_type = matching_type.split('_')[1]
        run_exhaustive_matcher(colmap_path, db_path, use_gpu,
                               colmap_matching_type)
    else:
        feature_name = feature_conf['output']
        pairs_path = results_path / 'paris-exhaustive.txt'
        pairs_from_exhaustive.main(pairs_path, features=feature_path)
        matching_conf = hloc_matching_confs[matching_type]
        match_path = match_features.main(matching_conf,
                                         pairs_path,
                                         feature_name,
                                         results_path)
        # create a empty database file
        db = COLMAPDatabase.connect(db_path)
        db.close()
        import_images(image_path, db_path, camera_mode='SINGLE')
        write_keypoints_into_db(db_path, feature_path)
        write_matches_into_db(db_path, match_path, pairs_path)
        # geometric verification via colmap matches_importer
        run_matches_importer(colmap_path, db_path, pairs_path, use_gpu,
                             geometric_verification_type)
    return
