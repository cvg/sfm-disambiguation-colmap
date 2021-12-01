import subprocess
import logging

from ..options.feature_options import colmap_sift_options
from ..options.matching_options import colmap_matching_options
from ..options.mapper_options import colmap_mapper_options


def run_feature_extractor(colmap_path, db_path, image_path, colmap_sift_type,
                          use_gpu):
    cmds = [
        f'{colmap_path}', 'feature_extractor', '--database_path', f'{db_path}',
        '--image_path', f'{image_path}'
    ]

    if isinstance(colmap_sift_type, dict):
        feature_options = colmap_sift_type
    else:
        assert colmap_sift_type in ['default', 'sparse']
        feature_options = colmap_sift_options[colmap_sift_type]

    if use_gpu:
        feature_options['--SiftExtraction.use_gpu'] = '1'
    else:
        feature_options['--SiftExtraction.use_gpu'] = '0'

    for key, value in feature_options.items():
        cmds.append(key)
        cmds.append(value)

    logging.info(' '.join(cmds))
    subprocess.run(cmds, check=True)
    return


def run_exhaustive_matcher(colmap_path, db_path, use_gpu,
                           colmap_matching_type):
    cmds = [
        f'{colmap_path}', 'exhaustive_matcher', '--database_path', f'{db_path}'
    ]

    if isinstance(colmap_matching_type, dict):
        matching_options = colmap_matching_type
    else:
        assert colmap_matching_type in ['default', 'strict']
        matching_options = colmap_matching_options[colmap_matching_type]

    if use_gpu:
        matching_options['--SiftMatching.use_gpu'] = '1'
    else:
        matching_options['--SiftMatching.use_gpu'] = '0'

    for key, value in matching_options.items():
        cmds.append(key)
        cmds.append(value)

    logging.info(' '.join(cmds))
    subprocess.run(cmds, check=True)
    return


def run_matches_importer(colmap_path, db_path, match_list_path, use_gpu,
                         colmap_matching_type):
    cmds = [
        f'{colmap_path}', 'matches_importer', '--database_path', f'{db_path}',
        '--match_list_path', f'{match_list_path}'
    ]

    if isinstance(colmap_matching_type, dict):
        matching_options = colmap_matching_type
    else:
        assert colmap_matching_type in ['default', 'strict']
        matching_options = colmap_matching_options[colmap_matching_type]

    if use_gpu:
        matching_options['--SiftMatching.use_gpu'] = '1'
    else:
        matching_options['--SiftMatching.use_gpu'] = '0'

    for key, value in matching_options.items():
        cmds.append(key)
        cmds.append(value)

    logging.info(' '.join(cmds))
    subprocess.run(cmds, check=True)
    return


def run_mapper(colmap_path, db_path, image_path, output_path,
               colmap_mapper_type):
    output_path.mkdir(exist_ok=True)
    cmds = [
        f'{colmap_path}', 'mapper', '--database_path', f'{db_path}',
        '--image_path', f'{image_path}', '--output_path', f'{output_path}'
    ]

    if isinstance(colmap_mapper_type, dict):
        mapper_options = colmap_mapper_type
    else:
        assert colmap_mapper_type in ['default', 'fix_intrinsics']
        mapper_options = colmap_mapper_options[colmap_mapper_type]

    for key, value in mapper_options.items():
        cmds.append(key)
        cmds.append(value)

    logging.info(' '.join(cmds))
    subprocess.run(cmds, check=True)
    return
