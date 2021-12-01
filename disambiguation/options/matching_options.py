colmap_matching_options = {
    'default': {},
    'strict': {
        "--SiftMatching.max_num_trials": "100000",
        "--SiftMatching.min_inlier_ratio": "0.05",
        "--SiftMatching.max_error": "2",
    },
}
hloc_matching_confs = {
    'nn': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'mutual_check': True,
            'distance_threshold': 0.7,
        }
    },
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    }
}
