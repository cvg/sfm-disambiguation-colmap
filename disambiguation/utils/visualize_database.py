import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from hloc.utils.viz import plot_images, plot_keypoints, plot_matches

from .database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
from .read_write_database import (read_image_names_from_db,
                                  read_keypoints_from_db)


def read_image(path):
    assert path.exists(), path
    image = cv2.imread(str(path))
    if len(image.shape) == 3:
        image = image[:, :, ::-1]
    return image


def visualize_keypoints(image_path,
                        db_path,
                        show=True,
                        save_path=None,
                        max_num=10):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path

    names = read_image_names_from_db(db)

    keypoints_results = db.execute("SELECT * FROM keypoints")
    num = 0
    for keypoints_result in keypoints_results:
        image_id, rows, cols, keypoints = keypoints_result
        keypoints = blob_to_array(keypoints, np.float32, (rows, cols))
        name = names[image_id]
        image = read_image(image_path / name)
        plot_images([image], titles=[name])
        plot_keypoints([keypoints])
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path / name)
        plt.close()
        num += 1
        if num >= max_num:
            break

    if is_path:
        db.close()
    return


def visualize_matches(image_path,
                      db_path,
                      show=True,
                      save_path=None,
                      max_num=10):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path

    names = read_image_names_from_db(db)
    kpts = read_keypoints_from_db(db)

    matches_results = db.execute("SELECT * FROM matches")
    num = 0
    for matches_result in matches_results:
        pair_id, rows, cols, matches = matches_result
        if rows == 0:
            continue
        image_id0, image_id1 = pair_id_to_image_ids(pair_id)
        name0 = names[image_id0]
        name1 = names[image_id1]
        image0 = read_image(image_path / name0)
        image1 = read_image(image_path / name1)
        kpts0 = kpts[image_id0]
        kpts1 = kpts[image_id1]
        matches = blob_to_array(matches, dtype=np.uint32, shape=(rows, cols))
        matched_kpts0 = []
        matched_kpts1 = []
        for kpts0_id, kpts1_id in matches:
            matched_kpts0.append(kpts0[kpts0_id, :2])
            matched_kpts1.append(kpts1[kpts1_id, :2])
        matched_kpts0 = np.stack(matched_kpts0, axis=0)
        matched_kpts1 = np.stack(matched_kpts1, axis=0)
        plot_images(imgs=[image0, image1], titles=[name0, name1])
        plot_matches(matched_kpts0, matched_kpts1, a=0.2)
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path / f'{name0}_{name1}.png')
        plt.close()
        num += 1
        if num >= max_num:
            break

    if is_path:
        db.close()
    return


def visualize_inlier_matches(image_path,
                             db_path,
                             show=True,
                             save_path=None,
                             max_num=10):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path

    names = read_image_names_from_db(db)
    kpts = read_keypoints_from_db(db)

    inlier_matches_results = db.execute(
        "SELECT pair_id, rows, cols, data FROM two_view_geometries")
    num = 0
    for inlier_matches_result in inlier_matches_results:
        pair_id, rows, cols, matches = inlier_matches_result
        if rows == 0:
            continue
        image_id0, image_id1 = pair_id_to_image_ids(pair_id)
        name0 = names[image_id0]
        name1 = names[image_id1]
        image0 = read_image(image_path / name0)
        image1 = read_image(image_path / name1)
        kpts0 = kpts[image_id0]
        kpts1 = kpts[image_id1]
        matches = blob_to_array(matches, dtype=np.uint32, shape=(rows, cols))
        matched_kpts0 = []
        matched_kpts1 = []
        for kpts0_id, kpts1_id in matches:
            matched_kpts0.append(kpts0[kpts0_id, :2])
            matched_kpts1.append(kpts1[kpts1_id, :2])
        matched_kpts0 = np.stack(matched_kpts0, axis=0)
        matched_kpts1 = np.stack(matched_kpts1, axis=0)
        plot_images(imgs=[image0, image1], titles=[name0, name1])
        plot_matches(matched_kpts0, matched_kpts1, a=0.2)
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path / f'{name0}_{name1}_inlier.png')
        plt.close()
        num += 1
        if num >= max_num:
            break

    if is_path:
        db.close()
    return
