#-*- coding: utf-8 -*-

import os
import datetime
import random
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from parse import parse_args
from torch import sparse
from tqdm import tqdm
from data import *
from model import *
from evaluation import *
from negative_sampling import *
from cuda import *
from sklearn.cluster import KMeans, DBSCAN
# print(torch.__version__)
USE_CUDA = torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'

device = torch.device('cuda' if USE_CUDA else 'cpu')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_path():
    directory = 'data/'
    if arg.dataset == '100k':
        total_file = directory + '/' + '100k.csv'
        train_file = directory + '/' + '100k_train.csv'
        test_file = directory + '/' + '100k_test.csv'
    elif arg.dataset == '1M':
        total_file = directory + '/' + '1m1.csv'
        train_file = directory + '/' + '1m1_train.csv'
        test_file = directory + '/' + '1m1_test.csv'
    elif arg.dataset == 'gowalla':
        total_file = directory + '/' + 'gowalla.csv'
        train_file = directory + '/' + 'gowalla_train.csv'
        test_file = directory + '/' + 'gowalla_test.csv'
    elif arg.dataset == 'yelp2018':
        total_file = directory + '/' + 'yelp2018.csv'
        train_file = directory + '/' + 'yelp2018_train.csv'
        test_file = directory + '/' + 'yelp2018_test.csv'
    return total_file, train_file, test_file


def log():
    if arg.log:
        path = arg.log_root
        #path = arg.log_root + arg.dataset
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + '/' + arg.dataset + '_' + arg.LOSS + '_'+ arg.encoder + '_' + str(arg.M) + '_' + str(arg.N) + '_' + str(arg.tau_plus) + '--' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
        f = open(file, 'w')
        print('----------------loging----------------')
    else:
        f = sys.stdout
    return f


def get_numbers_of_ui_and_divider(file):
    '''
    :param file: data path
    :return:
    num_users: total number of users
    num_items: total number of items
    dividing_tensor: [|I|] element 1 represents hot items, while element 0 represents cold items.
    '''
    data = pd.read_csv(file, header=0, dtype='str', sep=',')
    userlist = list(data['user'].unique())
    itemlist = list(data['item'].unique())
    popularity = np.zeros(len(itemlist))
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
    num_users, num_items = len(userlist), len(itemlist)
    return num_users, num_items


def load_train_data(path, num_item):
    data = pd.read_csv(path, header=0, sep=',')
    datapair = []
    popularity = np.zeros(num_item)
    popularity_u = np.zeros(num_users)
    user_dict_list = {}
    item_dict_list = {}
    train_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        popularity_u[int(user)] += 1
        datapair.append((user, item))
        train_tensor[user, item] = 1
        if user in user_dict_list:
            user_dict_list[user].append(item)
        else:
            user_dict_list[user] = [item]

        if item in item_dict_list:
            item_dict_list[item].append(user)
        else:
            item_dict_list[item] = [user]

    prior = popularity / sum(popularity)
    prior_u = popularity_u / sum(popularity_u)
    return train_tensor.to_sparse(), prior, datapair, user_dict_list, item_dict_list, prior_u


def load_test_data(path, num_user, num_item):
    data = pd.read_csv(path, header=0, sep=',')
    test_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        test_tensor[user, item] = 1
    return test_tensor.bool()


def collect_G_Lap_Adj():
    G_Lap_tensor = convert_spmat_to_sptensor(dataset.Lap_mat)
    G_Adj_tensor = convert_spmat_to_sptensor(dataset.Adj_mat)
    G_Lap_tensor = G_Lap_tensor.to(device)
    G_Adj_tensor = G_Adj_tensor.to(device)
    return G_Lap_tensor, G_Adj_tensor


def model_init():
    # A new train
    model_path = r'.\model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if arg.train_mode == 'new_train':
        if arg.encoder == 'MF':
            model = MF(num_users, num_items, arg, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
        checkpoint = 0
    # Continue train
    else:
        checkpoint = torch.load(r'.\model\{}-{}-ex_model.pth'.format(arg.dataset, arg.encoder))
        if arg.encoder == 'MF':
            model = MF(num_users, num_items, arg, device)
        model.load_state_dict(checkpoint['net'])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('epoch_begin:', checkpoint['epoch'] + 1)
    return model, optimizer, scheduler, checkpoint


def model_train(real_epoch):
    print('-------------------------------------------', file=f)
    print('-------------------------------------------')
    print('epoch: ', real_epoch, file=f)
    print('epoch: ', real_epoch)
    print('start training: ', datetime.datetime.now(), file=f)
    print('start training: ', datetime.datetime.now())
    st = time.time()
    model.train()
    total_loss = []

    # Two tower model training
    for index, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        # To device
        batch = batch.to(device)
        # Fetch Data
        users = batch[:,0] #[bs,]
        positives = batch[:, 1 : arg.M+1 ] # [bs * M]
        negatives = batch[:, arg.M+1:]     # [bs * N]
        # Calculate Loss
        loss = model(users, positives, negatives, real_epoch)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    # LG2CF module before next epoch's encoding step
    if real_epoch >= arg.N_p:
        LG2CF()



    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Training time:[%0.2f s]' % (time.time() - st))
    print('Training time:[%0.2f s]' % (time.time() - st), file=f)


def model_test():
    print('----------------', file=f)
    print('----------------')
    print('start evaluation: ', datetime.datetime.now(), file=f)
    print('start evaluation: ', datetime.datetime.now())
    model.eval()

    Pre_dic, Recall_dict, F1_dict, NDCG_dict = {}, {}, {}, {}
    sp = time.time()
    rating_mat = model.predict() # |U| * |V|
    rating_mat = erase(rating_mat)
    for k in arg.topk:
        metrices = topk_eval(rating_mat, k, test_tensor.to(device))
        precision, recall, F1, ndcg = metrices[0], metrices[1], metrices[2], metrices[3]
        Pre_dic[k] = precision
        Recall_dict[k] = recall
        F1_dict[k] = F1
        NDCG_dict[k] = ndcg
    print('Evaluation time:[%0.2f s]' % (time.time() - sp))
    print('Evaluation time:[%0.2f s]' % (time.time() - sp), file=f)
    return Pre_dic, Recall_dict, F1_dict, NDCG_dict, rating_mat


def erase(score):
    x = train_tensor.to(device) * (-1000)
    score = score + x
    return score


def print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict, rating_mat):
    if Pre_dic[5] > best_result[5][0]:
        user_emb = np.array(model.User_Emb.weight.data.cpu().detach())
        item_emb = np.array(model.Item_Emb.weight.data.cpu().detach())
        np.save('best_user_emb.npy', user_emb)
        np.save('best_item_emb.npy', item_emb)
        topk_tensor = torch.topk(rating_mat, k=5, dim=1).indices
        rec_tensor = get_rec_tensor(5, topk_tensor, rating_mat.shape[1])
        np.save('topk_Proposed', np.array(rec_tensor.float().cpu().detach()))
    for k in arg.topk:
        if Pre_dic[k] > best_result[k][0]:
            best_result[k][0], best_epoch[k][0] = Pre_dic[k], real_epoch
        if Recall_dict[k] > best_result[k][1]:
            best_result[k][1], best_epoch[k][1] = Recall_dict[k], real_epoch
        if F1_dict[k] > best_result[k][2]:
            best_result[k][2], best_epoch[k][2] = F1_dict[k], real_epoch
        if NDCG_dict[k] > best_result[k][3]:
            best_result[k][3], best_epoch[k][3] = NDCG_dict[k], real_epoch
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k]))
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k]),
            file=f)
    return best_result, best_epoch


def print_best_result(best_result, best_epoch):
    print('------------------best result-------------------', file=f)
    print('------------------best result-------------------')
    for k in arg.topk:
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3],  (time.time() - t0)))
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3],  (time.time() - t0)), file=f)

        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3],
                (time.time() - t0)))
        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3],
                (time.time() - t0)), file=f)
    print('------------------------------------------------', file=f)
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0), file=f)
    print('Run time: %0.2f s' % (time.time() - t0))


def LG2CF():
    # Fetch original embeddings
    user_original_embedding = np.array(model.User_Emb.weight.data.cpu().detach())
    item_original_embedding = np.array(model.Item_Emb.weight.data.cpu().detach())

    # Clustering
    DBSCAN_i = DBSCAN(eps=0.5, min_samples=5).fit(item_original_embedding)
    DBSCAN_u = DBSCAN(eps=0.5, min_samples=5).fit(user_original_embedding)

    # Obtain latent groups
    item_cluster = DBSCAN_i.labels_
    user_cluster = DBSCAN_u.labels_
    num_DBSCAN_i = max(item_cluster) + 2  # label of clusters in DBSCAN starts from -1
    num_DBSCAN_u = max(user_cluster) + 2  # label of clusters in DBSCAN starts from -1

    # Average pooling to get latent group embeddings
    item_latent_group_embeddings = np.zeros((num_DBSCAN_i, arg.dim))
    centres_u = np.zeros((num_DBSCAN_u, arg.dim))
    for i in range(-1, num_DBSCAN_i - 1):
        item_latent_group_embeddings[i] = np.mean(item_original_embedding[item_cluster == i])
    for j in range(-1, num_DBSCAN_u - 1):
        centres_u[j] = np.mean(user_original_embedding[user_cluster == j])

    # Calculate IIG embeddings
    IIG = torch.zeros((num_users, arg.dim))
    for u in range(num_users):  # For each user
        # Fetch user's interacted items
        item_interacted = train_dict[u]
        # Fetch interacted items' label
        iteracted_item_label = item_cluster[item_interacted]
        # Calculate beta_j
        pop_label = pop[item_interacted] / pop[item_interacted].sum()
        # Weighted fusion
        IIG[u] = (torch.tensor(item_latent_group_embeddings[iteracted_item_label]) * np.expand_dims(pop_label, -1)).to(device).sum(dim=0) / len(pop_label)

    # Calculate UIG embdeddings
    UIG = torch.zeros((num_items, arg.dim))
    for i in range(num_items):  # For each item
        if i in item_dict_list:
            # Fetch user's interacted items
            user_interacted = item_dict_list[i]
            # Fetch interacted users' label
            iteracted_user_label = user_cluster[user_interacted]
            # Calculate beta_i
            pop_label_u = pop_u[user_interacted] / pop_u[user_interacted].sum()
            # Weighted fusion
            UIG[i] = (torch.tensor(centres_u[iteracted_user_label]) * np.expand_dims(pop_label_u, -1)).to(device).sum(dim=0) / len(pop_label_u)
        else:
            # Some items may have no interactions with users during the split approach
            UIG[i] = torch.mean(torch.tensor(centres_u).to(device), dim=0)

    # User and Item fusion
    item_id = range(0, num_items)
    user_id = range(0, num_users)
    model.User_Emb.weight.data[user_id] = arg.alpha * model.User_Emb.weight.data[user_id] + (1. - arg.alpha) * IIG.to(device)
    model.Item_Emb.weight.data[item_id] = arg.alpha * model.Item_Emb.weight.data[item_id] + (1. - arg.alpha) * UIG.to(device)


if __name__ == '__main__':
    t0 = time.time()
    arg = parse_args()
    f = log()
    print(arg)
    print(arg, file=f)
    init_seed(2022)
    total_file, train_file, test_file = get_data_path()
    num_users, num_items = get_numbers_of_ui_and_divider(total_file)

    # Load Data
    train_tensor, pop, train_pair, train_dict, item_dict_list, pop_u = load_train_data(train_file, num_items)
    test_tensor = load_test_data(test_file, num_users, num_items)

    dataset = Data(train_pair, arg, num_users, num_items)
    train_loader = DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, pin_memory=True, num_workers=arg.num_workers)

    # Init Model
    model, optimizer, scheduler, checkpoint = model_init()

    best_result = {}
    best_epoch = {}
    for k in arg.topk:
        best_result[k] = [0., 0., 0., 0.]
        best_epoch[k] = [0, 0, 0, 0]

    # Train and Test
    for epoch in range(arg.epochs):
        if arg.train_mode == 'new_train':
            real_epoch = epoch
        else:
            real_epoch = checkpoint['epoch'] + 1 + epoch
        model_train(real_epoch)
        # mc.show_cuda_info()
        Pre_dic, Recall_dict, F1_dict, NDCG_dict, rating_mat = model_test()

        scheduler.step()
        best_result, best_epoch = print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict, rating_mat)
    print_best_result(best_result, best_epoch)
    f.close()

    # Save Checkpoint
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': real_epoch}
    torch.save(state, r'.\model\{}-{}-ex_model.pth'.format(arg.dataset, arg.encoder))






