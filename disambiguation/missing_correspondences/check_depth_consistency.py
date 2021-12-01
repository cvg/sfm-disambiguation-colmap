import numpy as np


def check_depth_consistency(num_images, scales, depths_list):
    n = len(depths_list)
    median_depths = np.zeros((n,))
    num_valid_depths = np.zeros((num_images,), dtype=np.int)
    count = 0
    for i in range(n):
        m = len(depths_list[i])
        if m == 0:
            # use a minus value to indicate points not reconstructed
            median_depths[i] = -1.0
            # print(f"{i}th keypoint not reconstructed")
            continue
        count += 1
        id1s = np.zeros((m,), dtype=np.int)
        scaled_depths = np.zeros((m,))
        for j in range(m):
            id1s[j] = depths_list[i][j][0]
            scaled_depths[j] = depths_list[i][j][1] * scales[id1s[j]]
        median_depths[i] = np.median(scaled_depths)
        if median_depths[i] <= 0:
            # half of the reconstructed depths are wrong, remove this point
            median_depths[i] = -1.0
        for j in range(m):
            ratio = scaled_depths[j] / median_depths[i]
            if ratio >= 0.95 and ratio <= 1.05:
                num_valid_depths[id1s[j] - 1] += 1

    # logging.info(f"{count} of all {n} points are reconstructed")

    return median_depths, num_valid_depths
