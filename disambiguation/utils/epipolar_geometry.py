import numpy as np
import torch


def decompose_E(E, kpts0, kpts1, matches):
    '''
    E: essential matrix between two images
    kpts0: (N, 3) array of calibrated keypoint coordinates in image 0
    kpts1: (M, 3) array of calibrated keypoint coordinates in image 1
    matches: (K, 2) array of matched keypoints' indices
    '''
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vh = np.linalg.svd(E)

    t = U[:, 2]
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    P0 = np.identity(4)
    P1s = np.zeros((4, 4, 4))
    # 4 possible solutions
    P1s[:, 3, 3] = 1.0
    P1s[0, :3, :3] = R1
    P1s[0, :3, 3] = t
    P1s[1, :3, :3] = R1
    P1s[1, :3, 3] = -t
    P1s[2, :3, :3] = R2
    P1s[2, :3, 3] = t
    P1s[3, :3, :3] = R2
    P1s[3, :3, 3] = -t

    max_count = -1
    for i in range(4):
        p3ds_i, err_i = linear_triangulation(P0, kpts0, P1s[i], kpts1, matches)
        p2ds0 = (P0 @ p3ds_i.T).T
        p2ds1 = (P1s[i] @ p3ds_i.T).T

        count = np.sum(np.logical_and(p2ds0[:, 2] >= 0, p2ds1[:, 2] >= 0))
        if count > max_count:
            max_count = count
            P1 = P1s[i]
            p3ds = p3ds_i
            err = err_i
    if np.linalg.det(P1) < 0:
        P1[:3, :4] = -P1[:3, :4]

    # P1's translation's norm will be 1
    assert (np.isclose(np.linalg.norm(P1[:3, 3]), 1))
    assert p3ds.shape[1] == 4
    return P1, p3ds, err


def linear_triangulation(P0, kpts0, P1, kpts1, matches):
    '''
    P0: rotation & translation of the first camera,
        most likely to be I and 0. (4, 4)
    kpts0: calibrated keypoint coordinates in image 0, (N, 3)
    P1: rotation & translation of the second camera, (4, 4)
    kpts1: calibrated keypoint coordinates in image 1, (M, 3)
    matches: (K, 2) array of matched keypoints' indices
    '''
    K = matches.shape[0]
    p3ds = np.zeros((K, 4))
    err = np.zeros((K,))
    A = np.zeros((4, 4))
    # print(P1)
    for i in range(K):
        A[0] = kpts0[matches[i, 0], 0] * P0[2] - P0[0]
        A[1] = kpts0[matches[i, 0], 1] * P0[2] - P0[1]
        A[2] = kpts1[matches[i, 1], 0] * P1[2] - P1[0]
        A[3] = kpts1[matches[i, 1], 1] * P1[2] - P1[1]

        U, s, Vh = np.linalg.svd(A)
        p3ds[i] = Vh[3] / Vh[3, 3]
        err[i] = s[3]

    return p3ds, err


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / points[..., -1:]
