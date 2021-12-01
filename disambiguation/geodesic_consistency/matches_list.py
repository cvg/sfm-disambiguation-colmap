import numpy as np
import logging

from ..utils.database import COLMAPDatabase
from ..utils.database import pair_id_to_image_ids, blob_to_array


class MatchesList:
    """
    data structure to store matches_list.
    'dict' is fast but memory intensive.
    'smallarray' requires much less memory but is quite slow.
    'largearray' is a tradeoff between two data structures mentioned above.
    """

    def __init__(self, num_images, max_num_keypoints, ds, db_path):
        logging.info("----------Create MatchesList----------")
        assert ds in [
            'dict', 'smallarray', 'largearray'
        ], "ds must be one of ['dict', 'smallarray', 'largearray']"
        self.ds = ds
        if ds == 'dict':
            self.matches_list = {}
        elif ds == 'smallarray':
            # `[[]] * N` create a list containing the same list object N times!
            # self.matches_list = [[]] * num_images
            self.matches_list = [[] for _ in range(num_images)]
        else:
            self.matches_list = np.full(
                (num_images, num_images, max_num_keypoints),
                -1,
                dtype=np.int32)
        self.neighbors_list = [[] for _ in range(num_images)]
        database = COLMAPDatabase.connect(db_path)
        matches_results = database.execute("select * FROM matches")
        for matches_result in matches_results:
            pair_id, rows, cols, matches = matches_result
            if rows == 0:
                continue
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            matches = blob_to_array(matches, np.uint32, (rows, cols))
            # image_id is 1-based
            self.neighbors_list[image_id1 - 1].append(image_id2)
            self.neighbors_list[image_id2 - 1].append(image_id1)
            if ds == 'dict':
                for keypoint_id1, keypoint_id2 in matches:
                    key1 = (image_id1, keypoint_id1)
                    key2 = (image_id2, keypoint_id2)
                    if key1 in self.matches_list:
                        self.matches_list[key1].append(key2)
                    else:
                        self.matches_list[key1] = [key2]
                    if key2 in self.matches_list:
                        self.matches_list[key2].append(key1)
                    else:
                        self.matches_list[key2] = [key1]
            elif ds == 'smallarray':
                matches_1 = self._sort_matches(matches)
                matches_2 = matches[:, [1, 0]]
                self.matches_list[image_id1 - 1].append((image_id2, matches_1))
                self.matches_list[image_id2 - 1].append((image_id1, matches_2))
            else:
                for keypoint_id1, keypoint_id2 in matches:
                    self.matches_list[image_id1 -
                                      1][image_id2 -
                                         1][keypoint_id1] = keypoint_id2
                    self.matches_list[image_id2 -
                                      1][image_id1 -
                                         1][keypoint_id2] = keypoint_id1

        database.close()
        logging.info("----------MatchesList Done----------")

    def __getitem__(self, idx):
        if self.ds == 'dict':
            if idx in self.matches_list:
                return self.matches_list[idx]
            else:
                return []
        elif self.ds == 'smallarray':
            return self.matches_list[idx]
        else:
            return self.matches_list[idx]

    def get_neighbors(self, idx):
        return self.neighbors_list[idx]

    def _sort_matches(self, matches):
        # matches (n, 2) array.
        # sort according to first column for binary search
        order = np.argsort(matches[:, 0])
        return matches[order, :]
