from pathlib import Path
import argparse
import glob

from disambiguation import (extract_match_features, set_logger)


def extract_and_match_features(feature_type, matching_type,
                               geometric_verification_type, colmap_path,
                               dataset_name, use_gpu):
    print('extract and match features: '
          f'{feature_type}, {matching_type}, {geometric_verification_type}')
    dataset_path = root / 'datasets' / dataset_folder / dataset_name
    results_path = (
        root / 'results' / dataset_name /
        '_'.join([feature_type, matching_type, geometric_verification_type]))
    results_path.mkdir(exist_ok=True, parents=True)
    db_path = results_path / (dataset_name + '.db')
    if db_path.is_file():
        return
    log_path = results_path / 'log.txt'
    set_logger(log_path)
    extract_match_features.main(feature_type,
                                matching_type,
                                geometric_verification_type,
                                dataset_path,
                                results_path,
                                colmap_path,
                                db_path,
                                use_gpu=use_gpu,
                                log_path=log_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--feature_type',
                        type=str,
                        required=True,
                        choices=[
                            'sift_default', 'sift_sparse', 'superpoint',
                            'd2net', 'r2d2', 'disk'
                        ])
    parser.add_argument(
        '--matching_type',
        type=str,
        required=True,
        choices=['sift_default', 'sift_strict', 'superglue', 'nn'])
    parser.add_argument('--geometric_verification_type',
                        type=str,
                        required=True,
                        choices=['default', 'strict'])
    parser.add_argument('--colmap_path', type=Path, default='colmap')
    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()
    print(args)

    root = Path(__file__).resolve().parents[1]
    dataset_path_heinly = root / 'datasets' / 'heinly2014'
    dataset_names_heinly = [
        path.split('/')[-1]
        for path in glob.glob(str(dataset_path_heinly / "*/"))
    ]
    dataset_path_yan = root / 'datasets' / 'yan2017'
    dataset_names_yan = [
        path.split('/')[-1] for path in glob.glob(str(dataset_path_yan / "*/"))
    ]
    if args.dataset_name in dataset_names_heinly:
        dataset_folder = 'heinly2014'
    elif args.dataset_name in dataset_names_yan:
        dataset_folder = 'yan2017'
    else:
        print(f"Unknown dataset name: {args.dataset_name}")
        raise ValueError

    extract_and_match_features(args.feature_type, args.matching_type,
                               args.geometric_verification_type,
                               args.colmap_path, args.dataset_name,
                               args.use_gpu)
