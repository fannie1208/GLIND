from data_utils import normalize


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--data_dir', type=str,
                        default='../../../OODgraph-GNNSafe/data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=500)

    # model network
    parser.add_argument('--method', type=str, default='erm')
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')

    #irm
    parser.add_argument('--irm_lambda', default=1,
                        type=float, help='lambda for irm')
    parser.add_argument('--irm_penalty_anneal_iter', default=500,
                        help='the step to reset optimizer in irm')

    parser.add_argument('--dann_alpha', type=float,
                        default=0.2, help='alpha for dann')
    parser.add_argument('--coral_penalty_weight', type=float, default=0.001,
                        help='penalty weight for coral loss')
    parser.add_argument('--groupdro_step_size', type=float, default=0.01,
                        help='step size for groupdro')

    #mixup
    parser.add_argument('--mixup_prob', default=0.4, type=float,
                        help='prob for using batch mixup')
    parser.add_argument('--mixup_alpha', default=0.2, type=float,
                        help='alpha for beta distribution in mixup')
    parser.add_argument('--label_smooth_val', default=0.1, type=float,
                        help='the smooth value for label smooth loss in mixup')

    #srgnn
    parser.add_argument('--srgnn_alpha', type=float,
                        default=1, help='alpha for srgnn')
    parser.add_argument('--kmm_beta', type=float, default=0.2,
                        help='beta for calculating KMM weight.')

    #eerm
    parser.add_argument('--env_K', type=int, default=3,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=1,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--num_sample', type=int, default=5,
                        help='num of samples for each node with graph edit')
    parser.add_argument('--eerm_beta', type=float, default=2.0,
                        help='weight for mean of risks from multiple domains')
    parser.add_argument('--lr_a', type=float, default=0.005,
                        help='learning rate for graph edit model')

    #ours
    parser.add_argument('--backbone_type', type=str, default='gcn', choices=['gcn', 'sage', 'gat', 'difformer'])
    parser.add_argument('--K', type=int, default=3,
                        help='num of domains, each for one graph convolution filter')
    parser.add_argument('--tau', type=float, default=1,
                        help='temperature for Gumbel Softmax')
    parser.add_argument('--context_type', type=str, default='node', choices=['node', 'graph'])
    parser.add_argument('--prior', type=str, default='uniform', choices=['uniform', 'mixture'])
    parser.add_argument('--lamda', type=float, default=1.0,
                        help='weight for regularlization')
    parser.add_argument('--variant', action='store_true',help='set to use variant')
    parser.add_argument('--r', type=float, default=0.01,
                        help='ratio for mixture')

    # DPPIN
    parser.add_argument('--window', type=int, default=10, help='number of past timestamps')
    parser.add_argument('--train_num', type=int, default=4, help='number of training environment')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of training samples')
    parser.add_argument('--valid_ratio', type=float, default=0.8, help='ratio of validation samples')

    # GraphGPS
    parser.add_argument('--gps_heads', type=int, default=8, help='number of heads in GraphGPS')
    parser.add_argument('--attn_dropout', type=float, default=0, help='attention dropout in GraphGPS')

    # DIFFormer
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--num_heads', type=int, default=1,help='number of heads for attention')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])

    # training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--save_result', action='store_true', help='save results')

