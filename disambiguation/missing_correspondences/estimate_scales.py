import numpy as np

from .l1decode_pd import l1decode_pd
from ..utils.database import image_ids_to_pair_id, pair_id_to_image_ids


def estimate_djks(depths_list):
    djks = {}
    for i in range(len(depths_list)):
        for j in range(len(depths_list[i])):
            for k in range(j + 1, len(depths_list[i])):
                dj = depths_list[i][j]
                dk = depths_list[i][k]
                if dj[0] > dk[0]:
                    dj, dk = dk, dj
                pair_id = image_ids_to_pair_id(dj[0], dk[0])
                if pair_id in djks:
                    djks[pair_id].append(dj[1] / dk[1])
                else:
                    djks[pair_id] = [dj[1] / dk[1]]
    return djks


def estimate_scales(depths_list, neighbor_ids):
    djks = estimate_djks(depths_list)
    # filter out bad neighbors with no common points with any other neighbors
    valid_neighbor_ids_set = set()
    for pair_id in djks.keys():
        id0, id1 = pair_id_to_image_ids(pair_id)
        valid_neighbor_ids_set.add(id0)
        valid_neighbor_ids_set.add(id1)

    valid_neighbor_ids = []
    invalid_neighbor_ids = []
    for neighbor_id in neighbor_ids:
        if neighbor_id in valid_neighbor_ids_set:
            valid_neighbor_ids.append(neighbor_id)
        else:
            invalid_neighbor_ids.append(neighbor_id)

    num_lp_vars = len(valid_neighbor_ids)
    if num_lp_vars == 0:
        return None, None

    image_id_to_var_id = {}
    var_id_to_image_id = {}
    for i, neighbor_id in enumerate(valid_neighbor_ids):
        image_id_to_var_id[neighbor_id] = i
        var_id_to_image_id[i] = neighbor_id

    A = np.zeros((len(djks) + 1, num_lp_vars))
    b = np.zeros((len(djks) + 1, 1))
    x0 = np.zeros((num_lp_vars, 1))

    # Cui ICCV 2015 paper fomula (5)
    for i, pair_id in enumerate(djks.keys()):
        id0, id1 = pair_id_to_image_ids(pair_id)
        A[i, image_id_to_var_id[id0]] = -1
        A[i, image_id_to_var_id[id1]] = 1
        b[i] = np.log(np.median(djks[pair_id]))

    # enforece the scale of the neighbor with max #matches to be 1
    # the neighbors are sorted so its index will be 0
    A[len(djks), 0] = 1
    b[len(djks)] = 0
    xp = l1decode_pd(x0, A, None, b)
    if xp is None:
        return None, None
    xp = np.exp(xp)
    assert xp.min() > 0
    scales = {}
    for i in range(xp.shape[0]):
        scales[var_id_to_image_id[i]] = xp[i]

    return scales, invalid_neighbor_ids
