import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Log Information.
    parser.add_argument('--log_root', default=r'.\exam_result')
    parser.add_argument('--log', type=bool, default=True)

    # Random Seed
    parser.add_argument('--seed', type=int, default=2022)

    # Training Args
    parser.add_argument('--train_mode', default='new_train', help='training mode:new_train continue_train')
    parser.add_argument('--encoder', default='MF', help='MF LightGCN')

    parser.add_argument('--epochs', type=int, default=1000)  # 1000
    parser.add_argument('--batch_size', type=int, default=1024)  # 128 1024
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # 1e-4 1e-5 0
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')  # 5e-4
    parser.add_argument('--lr_dc', type=float, default=1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_epoch', type=list, default=[100], help='the epoch which the learning rate decay')
    parser.add_argument('--LOSS', default='Info_NCE', help='loss')  # BPR, Info_NCE
    parser.add_argument('--num_workers', type=int, default=0)  # Speed up training
    parser.add_argument('--dim', type=int, default=32, help='dimension of vector')  # Dim of encoders
    parser.add_argument('--temperature', type=float, default=1, help='temperature')  # 0.5 1
    parser.add_argument('--tau_plus', type=float, default=0.07, help='tau_plus')  # 0.5 1

    # LG2CF args
    parser.add_argument('--N_p', type=int, default=50, help='number of pre-trained epoch')  # 150 200 300
    parser.add_argument('--alpha', type=float, default=0.987, help='alpha')

    # Dataset
    parser.add_argument('--dataset', default='100k', help='dataset')  # 100k 1M gowalla yelp2018

    # Sampling Args
    parser.add_argument('--M', type=int, default=5, help='number of positive samples for each user')
    parser.add_argument('--N', type=int, default=5, help='number of negative samples for each user')
    parser.add_argument('--C', type=int, default=10, help='number of negative samples for each user')
    parser.add_argument('--cluster', type=int, default=2, help='number of negative samples for each user')

    # Evaluation Arg
    parser.add_argument('--topk', type=list, default=[5, 10, 20], help='length of recommendation list')

    return parser.parse_args()


