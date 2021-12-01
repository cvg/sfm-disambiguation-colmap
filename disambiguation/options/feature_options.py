colmap_sift_options = {
    'default': {},
    'sparse': {
        '--SiftExtraction.first_octave': '0'
    }
}
hloc_feature_confs = {
    'superpoint': {
        'output': 'feats-superpoint-n8192-r2400',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 8192
        },
        'preprocessing': {
            # 'grayscale': False,
            'grayscale': True,
            'resize_max': 2400,
        }
    },
    'd2net': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 2400,
        }
    },
    'r2d2': {
        'output': 'feats-r2d2-top5000-max2400',
        'model': {
            'name': 'r2d2',
            'checkpoint_name': 'r2d2_WASF_N16.pt',
            'top-k': 5000,
            'scale-f': 2**0.25,
            'min-size': 256,  # 256,
            'max-size': 2400,  # 1024,
            'min-scale': 0,
            'max-scale': 1,
            'reliability-thr': 0.7,
            'repetability-thr': 0.7,
        },
        'preprocessing': {
            'resize_max': 2400,
        }
    },
    'disk': {
        'output': 'feats-disk-depth128-nms-window5',
        'model': {
            'name': 'disk',
            # Name of the model's .pth save file
            'model_name': 'depth-save.pth',
            # rescaled height (px).
            # If unspecified, image is not resized in height dimension
            # 'height': None,
            # rescaled width (px).
            # If unspecified, image is not resized in width dimension
            # 'width': None,
            # NMS window size
            'window': 5,
            # Maximum number of features to extract.
            # If unspecified, the number is not limited
            # 'n': None,
            'n': 25600,
            # descriptor dimension. Needs to match the checkpoint value
            'desc-dim': 128,
            # Whether to extract features using the non-maxima suppresion mode or through training-time grid sampling technique
            'mode': 'nms',
        },
        'preprocessing': {
            'resize_max': 2400,
        }
    }
}
