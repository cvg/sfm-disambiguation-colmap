import numpy as np
import logging
from pathlib import Path

from ..utils.database import COLMAPDatabase
from ..utils.database import pair_id_to_image_ids, blob_to_array
from ..utils.epipolar_geometry import decompose_E
from ..utils.epipolar_geometry import from_homogeneous, to_homogeneous


def reconstruct_depths(db_path,
                       center_id,
                       intrinsics,
                       max_num_neighbors=80,
                       use_E=False):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path
    # extract center image's keypoints
    rows0, cols0, kpts0 = next(
        db.execute("SELECT rows, cols, data FROM keypoints "
                   f"WHERE image_id = {center_id}"))
    kpts0 = blob_to_array(kpts0, np.float32, (rows0, cols0))[:, :2]
    kpts0_h = to_homogeneous(kpts0)
    K0 = intrinsics[center_id]
    kpts0_c = from_homogeneous((np.linalg.inv(K0) @ kpts0_h.T).T)

    # TODO: better way to store this data structure?
    depths_list = [[] for i in range(rows0)]
    num_matches_list = []
    # image_id_to_var_id = {}
    # var_id_to_image_id = {}
    # num_lp_vars = 0
    neighbor_ids = []
    poses = {}
    kpts0_matches = {}

    two_view_geometries = db.execute(
        "SELECT pair_id, rows FROM two_view_geometries")
    for pair_id, rows in two_view_geometries:
        id0, id1 = pair_id_to_image_ids(pair_id)
        if id0 == center_id or id1 == center_id:
            num_matches_list.append((pair_id, rows))
    num_matches_list.sort(key=lambda x: x[1], reverse=True)
    num_matches_list = num_matches_list[:max_num_neighbors]
    for pair_id, rows1 in num_matches_list:
        two_view_geometry = db.execute("SELECT cols, data, F, E "
                                       "FROM two_view_geometries "
                                       f"WHERE pair_id={pair_id}")
        cols1, matches, F, E = next(two_view_geometry)
        assert cols1 == 2
        id0, id1 = pair_id_to_image_ids(pair_id)
        if rows1 == 0:
            continue
        matches = blob_to_array(matches, np.uint32, (rows1, cols1))
        K1 = intrinsics[id1]

        if use_E:
            if E is None:
                logging.info("E is None! Skip it")
                continue
            E = blob_to_array(E, np.float64, (3, 3))
        else:
            if F is None:
                logging.info("F is None! Skip it")
                continue
            F = blob_to_array(F, np.float64, (3, 3))
            E = K1.T @ F @ K0

        if id1 == center_id:
            id0, id1 = id1, id0
            E = E.T
            matches = matches[:, ::-1]

        rows1, cols1, kpts1 = next(
            db.execute("SELECT rows, cols, data FROM keypoints "
                       f"WHERE image_id = {id1}"))
        kpts1 = blob_to_array(kpts1, np.float32, (rows1, cols1))[:, :2]
        kpts1_h = to_homogeneous(kpts1)
        kpts1_c = from_homogeneous((np.linalg.inv(K1) @ kpts1_h.T).T)

        P1, p3ds, _ = decompose_E(E, kpts0_c, kpts1_c, matches)
        mask = filter_invalid_3d_points(K0, K1, P1, p3ds)
        poses[id1] = P1
        for j in range(matches.shape[0]):
            if mask[j]:
                depths_list[matches[j, 0]].append((id1, p3ds[j, 2]))

        # some neighbors doesn't share a single point with any other neighbors
        # leading to a degenerative linear system
        # the fix is to return a temporary list of neighbors now
        # and get the final mapping after filtering out all bad neighbors
        neighbor_ids.append(id1)
        # image_id_to_var_id[id1] = num_lp_vars
        # var_id_to_image_id[num_lp_vars] = id1
        # num_lp_vars += 1
        kpts0_matches[id1] = matches[:, 0]

    if is_path:
        db.close()
    # return kpts0_c, depths_list, num_lp_vars, \
    #     image_id_to_var_id, var_id_to_image_id, kpts0_matches, poses
    return kpts0_c, depths_list, neighbor_ids, kpts0_matches, poses


def filter_invalid_3d_points(K0, K1, P1, p3ds):
    min0x = -K0[0, 2] / K0[0, 0]
    max0x = K0[0, 2] / K0[0, 0]
    min0y = -K0[1, 2] / K0[1, 1]
    max0y = K0[1, 2] / K0[1, 1]
    min1x = -K1[0, 2] / K1[0, 0]
    max1x = K1[0, 2] / K1[0, 0]
    min1y = -K1[1, 2] / K1[1, 1]
    max1y = K1[1, 2] / K1[1, 1]
    # P0 is identity
    p3ds_0 = p3ds
    p3ds_1 = (P1 @ p3ds.T).T
    mask0 = np.logical_and(p3ds_0[:, 0] / p3ds_0[:, 2] >= min0x,
                           p3ds_0[:, 0] / p3ds_0[:, 2] <= max0x)
    mask1 = np.logical_and(p3ds_0[:, 1] / p3ds_0[:, 2] >= min0y,
                           p3ds_0[:, 1] / p3ds_0[:, 2] <= max0y)
    mask2 = np.logical_and(p3ds_1[:, 0] / p3ds_1[:, 2] >= min1x,
                           p3ds_1[:, 0] / p3ds_1[:, 2] <= max1x)
    mask3 = np.logical_and(p3ds_1[:, 1] / p3ds_1[:, 2] >= min1y,
                           p3ds_1[:, 1] / p3ds_1[:, 2] <= max1y)
    mask4 = p3ds_0[:, 2] > 0
    mask5 = p3ds_1[:, 2] > 0
    mask = np.logical_and.reduce([mask0, mask1, mask2, mask3, mask4, mask5])
    return mask
