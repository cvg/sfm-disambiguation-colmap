import subprocess
from pathlib import Path
import numpy as np
import h5py
import logging

from ..utils.database import pair_id_to_image_ids, blob_to_array
from ..utils.database import COLMAPDatabase


def remove_matches_from_db(old_db_path, new_db_path, match_list_path, valid):
    """
    Args:
        old_db_path: Path object for the original database
        new_db_path: Path object for the new database
                     with invalid matches removed
        match_list_path: a txt file with two image file names each row.
                         each row is a valid match in the new database.
                         used for colmap matches_importer to compute
                         the table two_view_geometries
        valid: a 0-indexed (#images, #images) boolean array indicating whether
               one match should be preserved in the new database.
    """
    assert old_db_path.is_file(), "the original database doesn't exist!"
    if new_db_path.is_file():
        subprocess.run(['rm', f'{new_db_path}'], check=True)
        logging.warning("new_db_path already exists -- will overwrite it.")
    # copy the databse and only modify table matches/two_view_geometries
    subprocess.run(['cp', f'{old_db_path}', f'{new_db_path}'], check=True)
    new_db = COLMAPDatabase.connect(new_db_path)

    # Get a mapping between image ids and image names
    image_id_to_name = read_image_names_from_db(new_db)

    # remove matches
    matches_results = new_db.execute("SELECT pair_id, rows FROM matches")
    # write match_list.txt for matches_importer
    match_list_file = open(match_list_path, 'wt')

    for matches_result in matches_results:
        pair_id, rows = matches_result
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        if rows == 0:
            continue
        # if not (valid[image_id1 - 1][image_id2 - 1] and
        #         valid[image_id2 - 1][image_id1 - 1]):
        if not (valid[image_id1 - 1][image_id2 - 1] or
                valid[image_id2 - 1][image_id1 - 1]):
            # if not valid[image_id1 - 1][image_id2 - 1]:
            #     assert not valid[image_id2 - 1][image_id1 - 1]
            logging.info(f"matches between image {image_id1} and"
                         f" image {image_id2} are discarded")
            new_db.execute(f"DELETE FROM matches WHERE pair_id={pair_id}")
        else:
            # assert valid[image_id2 - 1][image_id1 - 1]
            match_list_file.write(f"{image_id_to_name[image_id1]} "
                                  f"{image_id_to_name[image_id2]}\n")

    # table two_view_geometries is deleted and will be computed by colmap
    new_db.execute("DROP TABLE two_view_geometries")

    match_list_file.close()
    new_db.commit()
    new_db.close()
    return


def write_keypoints_into_db(db_path, feature_path):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path
    image_name_to_id = read_image_ids_from_db(db)
    db.execute("DROP TABLE keypoints")
    db.create_keypoints_table()
    features_file = h5py.File(feature_path, 'r')
    for name in features_file:
        image_id = image_name_to_id[name]
        keypoints = features_file[name]['keypoints']
        db.add_keypoints(image_id=image_id, keypoints=keypoints)

    db.commit()
    if is_path:
        db.close()
    return


def write_matches_into_db(db_path, match_path, match_list_path):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path
    image_name_to_id = read_image_ids_from_db(db)
    match_file = h5py.File(match_path, 'r')
    db.execute("DROP TABLE matches")
    db.create_matches_table()
    with open(str(match_list_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]
    for name0, name1 in pairs:
        image_id0 = image_name_to_id[name0]
        image_id1 = image_name_to_id[name1]
        pair_name = names_to_pair(name0, name1)
        matches = match_file[pair_name]['matches0'][()]
        valid = matches > -1
        matches = np.stack([np.where(valid)[0], matches[valid]], -1)
        db.add_matches(image_id0, image_id1, matches)

    db.commit()
    if is_path:
        db.close()
    return


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def read_image_names_from_db(db_path):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path

    image_results = db.execute("SELECT image_id, name FROM images")
    image_id_to_name = {image_id: name for image_id, name in image_results}
    if is_path:
        db.close()
    return image_id_to_name


def read_keypoints_from_db(db_path):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path

    image_id_to_keypoints = {}
    keypoints_results = db.execute("SELECT * FROM keypoints")
    for keypoints_result in keypoints_results:
        image_id, rows, cols, keypoints = keypoints_result
        keypoints = blob_to_array(keypoints, np.float32, (rows, cols))
        image_id_to_keypoints[image_id] = keypoints

    if is_path:
        db.close()
    return image_id_to_keypoints


def read_image_ids_from_db(db_path):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path

    image_results = db.execute("SELECT image_id, name FROM images")
    image_name_to_id = {name: image_id for image_id, name in image_results}
    if is_path:
        db.close()
    return image_name_to_id


def read_intrinsics_from_db(db_path):
    is_path = isinstance(db_path, Path) or isinstance(db_path, str)
    if is_path:
        db = COLMAPDatabase.connect(db_path)
    else:
        assert isinstance(db_path, COLMAPDatabase)
        db = db_path
    images = db.execute("SELECT image_id, camera_id FROM images")
    intrinsics = {}
    for image in images:
        image_id, camera_id = image
        params = next(
            db.execute(
                f"SELECT params FROM cameras WHERE camera_id = {camera_id}")
        )[0]
        intrinsics[image_id] = params_to_intrinsics(
            blob_to_array(params, np.float64))
    if is_path:
        db.close()
    return intrinsics


def params_to_intrinsics(params):
    f, px, py, _ = params
    return np.array([[f, 0, px], [0, f, py], [0, 0, 1]])


def compare_databases(old_db_path, new_db_path):
    old_db = COLMAPDatabase.connect(old_db_path)
    new_db = COLMAPDatabase.connect(new_db_path)

    results1 = old_db.execute("SELECT * FROM cameras")
    results2 = new_db.execute("SELECT * FROM cameras")
    for result1, result2 in zip(results1, results2):
        camera_id1, model1, width1, height1, params1, prior1 = result1
        camera_id2, model2, width2, height2, params2, prior2 = result1
        params1 = blob_to_array(params1, np.float64)
        params2 = blob_to_array(params2, np.float64)
        assert camera_id1 == camera_id2
        assert model1 == model2
        assert width1 == width2
        assert height1 == height2
        assert np.all(params1 == params2)
        assert prior1 == prior2

    results1 = old_db.execute("SELECT * FROM descriptors")
    results2 = new_db.execute("SELECT * FROM descriptors")
    for result1, result2 in zip(results1, results2):
        image_id1, rows1, cols1, data1 = result1
        image_id2, rows2, cols2, data2 = result2
        data1 = blob_to_array(data1, np.uint8, (rows1, cols1))
        data2 = blob_to_array(data2, np.uint8, (rows2, cols2))
        assert image_id1 == image_id2
        assert np.all(data1 == data2)

    results1 = old_db.execute("SELECT * FROM images")
    results2 = new_db.execute("SELECT * FROM images")
    for result1, result2 in zip(results1, results2):
        image_id1, name1, camera_id1, prior_qw1, prior_qx1, prior_qy1, \
            prior_qz1, prior_tx1, prior_ty1, prior_tz1 = result1
        image_id2, name2, camera_id2, prior_qw2, prior_qx2, prior_qy2, \
            prior_qz2, prior_tx2, prior_ty2, prior_tz2 = result2
        assert image_id1 == image_id2
        assert name1 == name2
        assert camera_id1 == camera_id2
        assert prior_qw1 == prior_qw2
        assert prior_qx1 == prior_qx2
        assert prior_qy1 == prior_qy2
        assert prior_qz1 == prior_qz2
        assert prior_tx1 == prior_tx2
        assert prior_ty1 == prior_ty2
        assert prior_tz1 == prior_tz2

    results1 = old_db.execute("SELECT * FROM keypoints")
    results2 = new_db.execute("SELECT * FROM keypoints")
    for result1, result2 in zip(results1, results2):
        image_id1, rows1, cols1, keypoints1 = result1
        image_id2, rows2, cols2, keypoints2 = result2
        keypoints1 = blob_to_array(keypoints1, np.float32, (rows1, cols1))
        keypoints2 = blob_to_array(keypoints2, np.float32, (rows2, cols2))
        assert image_id1 == image_id2
        assert np.all(keypoints1 == keypoints2)

    logging.info(
        "Table cameras|descriptors|images|keypoints in the new database "
        "are the same as those in the original databse")

    # entries in the table matches in the new database
    # should be the subset of those in the old database
    # assuming the geometry verification is done under the same condition
    results2 = new_db.execute("SELECT * FROM matches")
    for result2 in results2:
        pair_id2, rows2, cols2, matches2 = result2
        if rows2 == 0:
            continue
        matches2 = blob_to_array(matches2, np.uint32, (rows2, cols2))
        result1 = next(
            old_db.execute(
                "SELECT * FROM matches WHERE pair_id = {}".format(pair_id2)))
        pair_id1, rows1, cols1, matches1 = result1
        matches1 = blob_to_array(matches1, np.uint32, (rows1, cols1))
        assert pair_id1 == pair_id2
        assert np.all(matches1 == matches2)

    logging.info("Table matches in the new database is "
                 "a subset of the that in the original databse")
    old_db.close()
    new_db.close()
