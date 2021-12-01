from pathlib import Path
import argparse
import glob

from disambiguation import (calculate_geodesic_consistency_scores, set_logger)


def disambiguate_via_geodesic_consistency(feature_type, matching_type,
                                          geometric_verification_type,
                                          dataset_name, track_degree,
                                          coverage_thres, alpha, minimal_views,
                                          ds):
    print('disambiguate via geodesic consistency: '
          f'{feature_type}, {matching_type}, {geometric_verification_type}')
    results_path = (
        root / 'results' / dataset_name /
        '_'.join([feature_type, matching_type, geometric_verification_type]))
    old_db_path = results_path / (dataset_name + '.db')
    log_path = results_path / 'log_yan.txt'
    set_logger(log_path)
    calculate_geodesic_consistency_scores.main(old_db_path, track_degree,
                                               coverage_thres, alpha,
                                               minimal_views, ds)
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

    parser.add_argument('--track_degree', type=int, default=3)
    parser.add_argument('--coverage_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--minimal_views', type=int, default=5)
    parser.add_argument('--ds',
                        type=str,
                        choices=['dict', 'smallarray', 'largearray'],
                        default='largearray')
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

    disambiguate_via_geodesic_consistency(args.feature_type,
                                          args.matching_type,
                                          args.geometric_verification_type,
                                          args.dataset_name, args.track_degree,
                                          args.coverage_thres, args.alpha,
                                          args.minimal_views, args.ds)
