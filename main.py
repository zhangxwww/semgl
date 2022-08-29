import argparse
import yaml
import torch
import numpy as np
# import far_ho as far
from datetime import datetime
import random
from collections import defaultdict, OrderedDict

from core.model_handler import ModelHandler

torch.autograd.set_detect_anomaly(True)

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config, args=None):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config, args)
    model.train()
    model.test()


# def multi_run_main(config, args=None):
#     print_config(config)
#     set_random_seed(config['random_seed'])
#     hyperparams = []
#     for k, v in config.items():
#         if isinstance(v, list):
#             hyperparams.append(k)

#     scores = []
#     configs = grid(config)
#     for cnf in configs:
#         print('\n')
#         now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#         rand = random.randint(0, 233)
#         for k in hyperparams:
#             cnf['out_dir'] += '_{}_{}_{}_{}'.format(k, cnf[k], now, rand)
#         print(cnf['out_dir'])
#         model = ModelHandler(cnf, args)
#         dev_metrics = model.train()
#         test_metrics = model.test()
#         scores.append(test_metrics[model.model.metric_name])

#     print('Average score: {}'.format(np.mean(scores)))
#     print('Std score: {}'.format(np.std(scores)))


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(args, config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    for k, v in args.items():
        config[k] = v
    config['learning_rate'] = args['lr']
    config['seed'] = config['random_seed']
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('-config',
                        '--config',
                        required=True,
                        type=str,
                        help='path to the config file')
    parser.add_argument('--multi_run',
                        action='store_true',
                        help='flag: multi run')
    parser.add_argument('--method',
                        type=str,
                        default='default',
                        help='which method to use')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--split_n', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--update_lambda1',
                        type=float,
                        default=1e-4,
                        help='which method to use')
    parser.add_argument('--update_lambda2',
                        type=float,
                        default=1e-1,
                        help='which method to use')
    parser.add_argument('--update_lambda3',
                        type=float,
                        default=0,
                        help='which method to use')
    parser.add_argument('--update_graph_alpha',
                        type=float,
                        default=5e-2,
                        help='which method to use')
    parser.add_argument('--alpha_decay',
                        type=float,
                        default=1,
                        help='which method to use')
    parser.add_argument('--alpha_decay_steps',
                        type=float,
                        default=5,
                        help='which method to use')
    parser.add_argument('--update_steps',
                        type=int,
                        default=1,
                        help='which method to use')
    parser.add_argument('--beta', type=float, default=1, help='learning rate')
    parser.add_argument('--relation_scale',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--update_detach_A',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--early_stop',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--eps_adj',
                        type=float,
                        default=1e-3,
                        help='learning rate')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--transductive',
                        action='store_false',
                        help='which method to use')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--ce_term',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--only_update_when_training',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--update_node_vec',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--smooth_gt_label',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--remain_pred',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--norm_A',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--large',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--two_gpu',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--test_armijo',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--update_lambda',
                        action='store_true',
                        help='which method to use')
    parser.add_argument('--random_seed', type=int)
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


# def grid(kwargs):
#     """Builds a mesh grid with given keyword arguments for this Config class.
#     If the value is not a list, then it is considered fixed"""
#     class MncDc:
#         """This is because np.meshgrid does not always work properly..."""
#         def __init__(self, a):
#             self.a = a  # tuple!

#         def __call__(self):
#             return self.a

#     sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
#     for k, v in sin.items():
#         copy_v = []
#         for e in v:
#             copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
#         sin[k] = copy_v

#     grd = np.array(np.meshgrid(*sin.values()),
#                    dtype=object).T.reshape(-1, len(sin.values()))
#     return [
#         far.utils.merge_dicts(
#             {k: v
#              for k, v in kwargs.items() if not isinstance(v, list)}, {
#                  k: vv[i]() if isinstance(vv[i], MncDc) else vv[i]
#                  for i, k in enumerate(sin)
#              }) for vv in grd
#     ]


################################################################################
# Module Command-line Behavior #
################################################################################

if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg, cfg['config'])
    # if cfg['multi_run']:
    #     multi_run_main(config, cfg)
    # else:
    main(config, cfg)
