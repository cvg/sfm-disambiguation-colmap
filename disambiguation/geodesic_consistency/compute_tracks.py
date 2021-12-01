import numpy as np
import queue
import logging


def compute_tracks(num_images, num_keypoints_list, matches_list, track_degree,
                   ds):
    """
        num_images: number of images in the database
        num_keypoints_list: list of #keypoints in each image (int)
        matches_list: matches_list[i] is a list of tuple (image_id, matches),
                      see class MatchesList
        track_degree: only consider tracks which appear in more than
                      `track_degree` images
        ds: indicate the data structure used for matches_list,
            either 'smallarray' or 'dict'
    """
    # these two flags are used for not generating inconsistent tracks
    # in which there are more than one feature in some views
    # in this way the track construction is consistent,
    # but not invariant to permutation
    # TODO: try filtering out inconsistent tracks
    # as done in the Bundler's commented code?
    logging.info("----------ComputeTracks Begins----------")
    img_marked = np.zeros((num_images,), dtype=np.bool)
    touched = []

    # a track is simply a list of (image_id, keypoint_id)
    # which marks the appearance of the 3D point in certain images
    tracks = []

    max_num_keypoints = max(num_keypoints_list)
    keypoints_visited = np.zeros((num_images, max_num_keypoints),
                                 dtype=np.bool)

    # image is 1-based indexed, keypoints is 0-based indexed
    # i here is 0-based so it can be used directly as index
    for i, num_keypoints in enumerate(num_keypoints_list):
        for j in range(num_keypoints):
            if keypoints_visited[i][j]:
                continue
            features = []
            features_queue = queue.Queue()

            # Reset flags
            for touched_idx in touched:
                img_marked[touched_idx - 1] = False
            touched = []

            # BFS on this keypoint
            keypoints_visited[i][j] = True

            # image_id is 1-based, keypoint_id is 0-based
            features.append((i + 1, j))
            features_queue.put((i + 1, j))

            img_marked[i] = True
            touched.append(i + 1)

            while not features_queue.empty():
                img_id1, keypoint_id1 = features_queue.get()

                if ds == 'dict':
                    for img_id2, keypoint_id2 in matches_list[(img_id1,
                                                               keypoint_id1)]:
                        # skip already visited images to avoid inconsistency
                        if img_marked[img_id2 - 1]:
                            continue
                        # already picked by other tracks
                        if keypoints_visited[img_id2 - 1, keypoint_id2]:
                            continue
                        img_marked[img_id2 - 1] = True
                        keypoints_visited[img_id2 - 1, keypoint_id2] = True
                        touched.append(img_id2)
                        features.append((img_id2, keypoint_id2))
                        features_queue.put((img_id2, keypoint_id2))
                elif ds == 'smallarray':
                    for img_id2, matches in matches_list[img_id1 - 1]:
                        if img_marked[img_id2 - 1]:
                            continue
                        match_id = np.searchsorted(matches[:, 0],
                                                   keypoint_id1,
                                                   side='left')
                        if match_id == matches.shape[0] or matches[
                                match_id, 0] != keypoint_id1:
                            continue
                        else:
                            keypoint_id2 = matches[match_id, 1]
                        assert keypoint_id2 < num_keypoints_list[img_id2 - 1]
                        if keypoints_visited[img_id2 - 1, keypoint_id2]:
                            continue
                        img_marked[img_id2 - 1] = True
                        keypoints_visited[img_id2 - 1, keypoint_id2] = True
                        touched.append(img_id2)
                        features.append((img_id2, keypoint_id2))
                        features_queue.put((img_id2, keypoint_id2))
                else:
                    matches = matches_list[img_id1 - 1][:, keypoint_id1]
                    # iterate over all image id
                    for k in np.where(matches != -1)[0]:
                        img_id2 = k + 1
                        if img_marked[img_id2 - 1]:
                            continue
                        keypoint_id2 = matches[img_id2 - 1]
                        if keypoints_visited[img_id2 - 1, keypoint_id2]:
                            continue
                        assert keypoint_id2 < num_keypoints_list[img_id2 - 1]
                        img_marked[img_id2 - 1] = True
                        keypoints_visited[img_id2 - 1, keypoint_id2] = True
                        touched.append(img_id2)
                        features.append((img_id2, keypoint_id2))
                        features_queue.put((img_id2, keypoint_id2))

            # found all features corresponding to one 3D point
            # from different views in a consistent way (by construction)
            if len(features) >= track_degree:
                # show up in enough number of images
                tracks.append(features)
                # logging.info(
                #     f"track {len(tracks)} consists of {len(features)} features"
                # )

    # All tracks have been computed
    # we check which tracks and keypoints are visible in each image
    # track_idx is 0-based
    visible_tracks = [[] for _ in range(num_images)]
    visible_keypoints = [[] for _ in range(num_images)]
    for track_idx, track in enumerate(tracks):
        for img_id, keypoint_id in track:
            visible_tracks[img_id - 1].append(track_idx)
            visible_keypoints[img_id - 1].append(keypoint_id)

    logging.info(f"{len(tracks)} tracks computed")
    logging.info("----------ComputeTracks Done----------")
    return tracks, visible_tracks, visible_keypoints
