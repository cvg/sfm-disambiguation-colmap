import h5py
import numpy as np
import matplotlib.pyplot as plt

from hloc.utils.viz import plot_images, plot_keypoints, plot_matches

from .visualize_database import read_image


def visualize_keypoints_h5(image_path,
                           feature_path,
                           show=True,
                           save_path=None):
    feature_file = h5py.File(feature_path, 'r')
    for name in feature_file.keys():
        image = read_image(image_path / name)
        kpts = feature_file[name]['keypoints'][()]
        plot_images(imgs=[image], titles=[name])
        plot_keypoints([kpts])
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

    feature_file.close()
    return


def visualize_matches_h5(image_path,
                         feature_path,
                         match_path,
                         show=True,
                         save_path=None):
    feature_file = h5py.File(feature_path, 'r')
    match_file = h5py.File(match_path, 'r')
    for name_pair in match_file.keys():
        name0, name1 = name_pair.split('_')
        image0 = read_image(image_path / name0)
        image1 = read_image(image_path / name1)
        kpts0 = feature_file[name0]['keypoints'][()]
        kpts1 = feature_file[name1]['keypoints'][()]
        matches = match_file[name_pair]['matches0'][()]
        matched_kpts0 = []
        matched_kpts1 = []
        for kpts0_id, kpts1_id in enumerate(matches):
            if kpts1_id != -1:
                matched_kpts0.append(kpts0[kpts0_id])
                matched_kpts1.append(kpts1[kpts1_id])
        matched_kpts0 = np.stack(matched_kpts0, axis=0)
        matched_kpts1 = np.stack(matched_kpts1, axis=0)
        plot_images(imgs=[image0, image1], titles=[name0, name1])
        plot_matches(matched_kpts0, matched_kpts1, a=0.2)
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

    feature_file.close()
    match_file.close()
    return
