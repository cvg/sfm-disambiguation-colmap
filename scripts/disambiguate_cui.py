from pathlib import Path
import argparse
import glob

from disambiguation import (calculate_missing_correspondences_scores,
                            set_logger)


def disambiguate_via_missing_correspondences(feature_type,
                                             matching_type,
                                             geometric_verification_type,
                                             dataset_name,
                                             min_num_valid_depths,
                                             max_num_neighbors,
                                             square_radius,
                                             score_version,
                                             parallel=True,
                                             plot=False):
    dataset_path = root / 'datasets' / dataset_folder / dataset_name
    image_path = dataset_path / 'images'
    results_path = (
        root / 'results' / dataset_name /
        '_'.join([feature_type, matching_type, geometric_verification_type]))
    results_path.mkdir(exist_ok=True, parents=True)
    old_db_path = results_path / (dataset_name + '.db')

    suffix = (f'_v{score_version}_d{min_num_valid_depths}' +
              f'_n{max_num_neighbors}_r{square_radius}')
    if parallel:
        suffix += '_p'
    if plot:
        plot_path = results_path / ('plots' + suffix)
    else:
        plot_path = None
    log_path = results_path / ('log' + suffix + '.txt')
    set_logger(log_path)
    calculate_missing_correspondences_scores.main(
        old_db_path,
        min_num_valid_depths=min_num_valid_depths,
        square_radius=square_radius,
        score_version=score_version,
        parallel=parallel,
        image_path=image_path,
        plot_path=plot_path)
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
    parser.add_argument('--score_version', type=int, choices=[1, 2, 3])
    parser.add_argument('--min_num_valid_depths', type=int, default=5)
    parser.add_argument('--max_num_neighbors', type=int, default=80)
    parser.add_argument('--square_radius', type=int, default=20)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--plot', action='store_true')
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

    disambiguate_via_missing_correspondences(
        args.feature_type, args.matching_type,
        args.geometric_verification_type, args.dataset_name,
        args.min_num_valid_depths, args.max_num_neighbors, args.square_radius,
        args.score_version, args.parallel, args.plot)
