import numpy as np
import logging
from matplotlib import pyplot as plt

from ..utils.visualize_database import read_image
from hloc.utils.viz import plot_images, plot_keypoints


def calculate_missing_scores_v2(num_images,
                                id0,
                                median_depths,
                                kpts0_c,
                                scales,
                                poses,
                                kpts0_matches,
                                intrinsics,
                                square_radius,
                                image_path=None,
                                image_names=None,
                                plot_path=None):
    assert square_radius > 0
    scores = np.zeros((num_images,))
    n = median_depths.shape[0]
    points = np.zeros((n, 4))
    points[:, 3] = 1
    for i in range(n):
        if median_depths[i] == -1:
            # use (0, 0, 0, 0) to represent points not reconstructed
            points[i, 3] = 0
            continue
        assert median_depths[i] > 0
        points[i, 0] = kpts0_c[i, 0] * median_depths[i]
        points[i, 1] = kpts0_c[i, 1] * median_depths[i]
        points[i, 2] = median_depths[i]

    visualize = image_path is not None and image_names is not None and \
        plot_path is not None
    if visualize:
        names = image_names
        plot_path.mkdir(exist_ok=True)

    K0 = intrinsics[id0]
    original_points = (K0 @ points[:, :3].T).T
    for id1 in poses.keys():
        P1 = poses[id1]
        P1[:3, 3] *= scales[id1]
        K1 = intrinsics[id1]
        matches = kpts0_matches[id1]
        projected_points = (K1 @ P1[:3] @ points.T).T
        missing_points = []
        matched_points = []
        colors = []
        kpts0_p = []
        kpts1_p = []
        for j in range(n):
            if original_points[j, 2] == 0:
                # points not reconstructed
                assert projected_points[j, 2] == 0
                continue
            if j in matches:
                matched_points.append(projected_points[j, :2] /
                                      projected_points[j, 2])
                colors.append((0., 1., 0))
            else:
                missing_points.append(projected_points[j, :2] /
                                      projected_points[j, 2])
                colors.append((1., 0., 0))
            kpts0_p.append(original_points[j, :2] / original_points[j, 2])
            kpts1_p.append(projected_points[j, :2] / projected_points[j, 2])
        matched_points = np.array(matched_points)
        missing_points = np.array(missing_points)
        pixel_min = np.array([[0, 0]])
        pixel_max = np.array([[K1[0, 2] * 2, K1[1, 2] * 2]])
        matched_inside_fov = np.logical_and(matched_points > pixel_min,
                                            matched_points < pixel_max)
        matched_inside_fov = np.logical_and(matched_inside_fov[:, 0],
                                            matched_inside_fov[:, 1])
        if len(missing_points) > 0:
            missing_inside_fov = np.logical_and(missing_points > pixel_min,
                                                missing_points < pixel_max)
            missing_inside_fov = np.logical_and(missing_inside_fov[:, 0],
                                                missing_inside_fov[:, 1])
            W, H = pixel_max.astype(np.int)[0]
            mask = np.zeros((H, W), dtype=np.bool)
            for i in range(matched_points.shape[0]):
                if matched_inside_fov[i]:
                    center = np.ceil(matched_points[i])
                    xc, yc = center.astype(np.int)
                    x_min = max(0, xc - square_radius)
                    x_max = min(W, xc + square_radius)
                    y_min = max(0, yc - square_radius)
                    y_max = min(H, yc + square_radius)
                    mask[y_min:y_max, x_min:x_max] = True

            missing_points_ceiled = np.floor(
                missing_points[missing_inside_fov]).astype(np.int)
            num_matched_missing_points = 0
            for i in range(missing_points_ceiled.shape[0]):
                x, y = missing_points_ceiled[i]
                if mask[y, x]:
                    num_matched_missing_points += 1
            score = num_matched_missing_points / np.sum(missing_inside_fov)
            logging.info(f'[{id0:4d}-{id1:4d}] '
                         f'inside square: {num_matched_missing_points:4d}; '
                         f'inside fov: {np.sum(missing_inside_fov):4d}; '
                         f'score: {score:.2f}')
        else:
            score = 1.
            logging.info(f"[{id0:4d}-{id1:4d}] all points matched; score: 1.0")
        scores[id1 - 1] = score

        if visualize:
            # image0 = read_image(image_path / names[id0])
            image1 = read_image(image_path / names[id1])
            kpts0_p = np.array(kpts0_p)
            kpts1_p = np.array(kpts1_p)
            inside = np.logical_and(kpts1_p >= pixel_min, kpts1_p <= pixel_max)
            inside = np.logical_and(inside[:, 0], inside[:, 1])
            # plot_images([image0, image1], [names[id0], names[id1]])
            # plot_matches(kpts0_p[inside],
            #              kpts1_p[inside],
            #              lw=0.5,
            #              color=[c for i, c in enumerate(colors) if inside[i]])
            # plt.savefig(plot_path / f"matches_{id0}_{id1}.png")
            # plot_images([image1], [names[id1]])
            # plot_keypoints([kpts1_p], colors=[colors])
            # plt.savefig(plot_path / f"kpts_all_{id0}_{id1}.png")
            colors = np.array(colors)
            plot_images([image1], [names[id1]])
            plot_keypoints([kpts1_p[inside]], colors=[colors[inside]])
            plt.savefig(plot_path / f"kpts_inside_{id0}_{id1}.png")
            plot_images([mask], [f'mask_{id0}_{id1}'])
            plot_keypoints([kpts1_p[inside]], colors=[colors[inside]])
            plt.savefig(plot_path / f"mask_{id0}_{id1}_{score:.2f}.png")
            plt.close('all')
    return scores
