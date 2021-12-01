from pathlib import Path
import argparse
import numpy as np
import glob
import networkx as nx
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree

from disambiguation.utils.read_write_database import remove_matches_from_db
from disambiguation.utils.run_colmap import run_matches_importer, run_mapper


def draw_graph(scores, plot_path, display=False):
    graph = nx.from_numpy_array(scores)
    # print(scores)
    pos = nx.nx_agraph.graphviz_layout(graph)
    edge_vmin = np.percentile(scores[scores.nonzero()], 10)
    edge_vmax = np.percentile(scores[scores.nonzero()], 90)
    # print(edge_vmin, edge_vmax)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        edge_color=[graph[u][v]['weight'] for u, v in graph.edges],
        # edge_cmap=plt.cm.plasma,
        edge_cmap=plt.cm.YlOrRd,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax)
    plt.savefig(plot_path)
    if display:
        plt.show()
    plt.close()
    return


def filter_with_fixed_threshold(scores, thres, plot_path=None):
    valid = scores >= thres
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=True)
    return valid


def filter_with_knn(scores, k, plot_path):
    valid = np.zeros_like(scores, dtype=np.bool)
    valid_indices = scores.argsort()[:, -k:]
    for i in range(scores.shape[0]):
        for j in valid_indices[i]:
            valid[i, j] = True
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=True)
    return valid


def filter_with_mst_min(scores, plot_path=None):
    min_scores = np.minimum(scores, scores.T)
    assert np.allclose(min_scores, min_scores.T)
    mst = minimum_spanning_tree(-min_scores)
    valid = (-mst).toarray() > 0
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=True)
    return valid


def filter_with_mst_mean(scores, plot_path=None):
    mean_scores = (scores + scores.T) / 2
    assert np.allclose(mean_scores, mean_scores.T)
    mst = minimum_spanning_tree(-mean_scores)
    valid = (-mst).toarray() > 0
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=True)
    return valid


def filter_with_percentile(scores, percentile, plot_path=None):
    num_images = scores.shape[0]
    thres = np.zeros((num_images, 1))
    for i in range(num_images):
        thres[i] = np.percentile(scores[i, scores[i].nonzero()], percentile)
    valid = scores >= thres
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=True)
    return valid


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

    parser.add_argument('--scores_name', type=str, required=True)
    parser.add_argument(
        '--filter_type',
        type=str,
        choices=['threshold', 'knn', 'mst_min', 'mst_mean', 'percentile'])
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--percentile', type=float)

    parser.add_argument('--reconstruct_unfiltered', action='store_true')
    parser.add_argument('--reconstruct_filtered', action='store_true')

    args = parser.parse_args()
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

    dataset_path = root / 'datasets' / dataset_folder / args.dataset_name
    image_path = dataset_path / 'images'
    results_path = (root / 'results' / args.dataset_name / '_'.join([
        args.feature_type, args.matching_type, args.geometric_verification_type
    ]))
    scores_path = results_path / args.scores_name
    scores = np.load(scores_path)
    if 'yan' in args.scores_name:
        method_name = 'yan'
    elif 'cui' in args.scores_name:
        method_name = 'cui'
    else:
        raise NotImplementedError

    # valid = scores >= args.threshold
    if args.filter_type == 'threshold':
        assert args.threshold is not None
        output_path = results_path / ('sparse' + args.scores_name[6:-4] +
                                      f'_t{args.threshold:.2f}')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        new_db_path = results_path / ('db' + args.scores_name[6:-4] +
                                      f'_t{args.threshold:.2f}.db')
        match_list_path = results_path / (
            'match_list' + args.scores_name[6:-4] + f'_t{args.threshold}.txt')
        valid = filter_with_fixed_threshold(scores, args.threshold, plot_path)
    elif args.filter_type == 'knn':
        assert args.topk is not None
        output_path = results_path / ('sparse' + args.scores_name[6:-4] +
                                      f'_k{args.topk}')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        new_db_path = results_path / ('db' + args.scores_name[6:-4] +
                                      f'_k{args.topk}.db')
        match_list_path = results_path / (
            'match_list' + args.scores_name[6:-4] + f'_k{args.topk}.txt')
        valid = filter_with_knn(scores, args.topk, plot_path)
    elif args.filter_type == 'percentile':
        assert args.percentile is not None
        output_path = results_path / ('sparse' + args.scores_name[6:-4] +
                                      f'_p{args.percentile}')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        new_db_path = results_path / ('db' + args.scores_name[6:-4] +
                                      f'_p{args.percentile}.db')
        match_list_path = results_path / (
            'match_list' + args.scores_name[6:-4] + f'_p{args.percentile}.txt')
        valid = filter_with_percentile(scores, args.percentile, plot_path)
    elif args.filter_type == 'mst_min':
        output_path = results_path / ('sparse' + args.scores_name[6:-4] +
                                      '_mst_min')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        new_db_path = results_path / ('db' + args.scores_name[6:-4] +
                                      '_mst_min.db')
        match_list_path = results_path / (
            'match_list' + args.scores_name[6:-4] + '_mst_min.txt')
        valid = filter_with_mst_min(scores, plot_path)
        # we don't do reconstruction based with mst graph as it is too sparse.
        # use it for visualization only
        exit(0)
    elif args.filter_type == 'mst_mean':
        output_path = results_path / ('sparse' + args.scores_name[6:-4] +
                                      '_mst_mean')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        new_db_path = results_path / ('db' + args.scores_name[6:-4] +
                                      '_mst_mean.db')
        match_list_path = results_path / (
            'match_list' + args.scores_name[6:-4] + '_mst_mean.txt')
        valid = filter_with_mst_mean(scores, plot_path)
        # we don't do reconstruction based with mst graph as it is too sparse.
        # use it for visualization only
        exit(0)
    else:
        raise NotImplementedError

    old_db_path = results_path / (args.dataset_name + '.db')
    remove_matches_from_db(old_db_path, new_db_path, match_list_path, valid)
    run_matches_importer(args.colmap_path,
                         new_db_path,
                         match_list_path,
                         use_gpu=False,
                         colmap_matching_type=args.geometric_verification_type)
    if args.reconstruct_unfiltered:
        run_mapper(args.colmap_path, old_db_path, image_path,
                   results_path / 'sparse', 'default')
    if args.reconstruct_filtered:
        run_mapper(args.colmap_path, new_db_path, image_path, output_path,
                   'default')
