# -*- coding: utf-8 -*-

import argparse
import math
import random
import string
from abc import ABC
from collections import defaultdict

import torch as th
from torch.nn import init

from data import MovieLens
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from util import *


def config():
    parser = argparse.ArgumentParser(description='RGC-ND-ED')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--model_save_path', type=str,
                        help='The model saving path')
    parser.add_argument('-dn', '--dataset_name', type=str,
                        help='dataset name')
    parser.add_argument('-dp', '--dataset_path', type=str,
                        help='raw dataset file path')

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=20)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--train_classification', type=bool, default=True)
    parser.add_argument('--review_feat_size', type=int, default=64)

    parser.add_argument('--rating_alpha', type=float, default=1.)
    parser.add_argument('--ed_alpha', type=float, default=.1)
    parser.add_argument('--nd_alpha', type=float, default=1.)
    parser.add_argument('--distributed', type=bool, default=False)

    args = parser.parse_args()
    args.model_short_name = 'RGC-ND-ED'

    args.dataset_name = 'Digital_Music_5'
    args.dataset_path = 'data/Digital_Music_5/Digital_Music_5.json'
    args.gcn_dropout = 0.7  # 0.7
    args.ed_alpha = 0.3  # 1.0
    args.nd_alpha = 0.1  # 0.3
    args.device = 0
    args.train_max_iter = 2000
    args.train_classification = False

    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')

    # configure save_fir to save all the info
    if args.model_save_path is None:
        args.model_save_path = 'log/' \
                               + args.model_short_name \
                               + '_' + args.dataset_name \
                               + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)) \
                               + '.pkl'
    if not os.path.isdir('log'):
        os.makedirs('log')

    args.gcn_agg_units = args.review_feat_size
    args.gcn_out_units = args.review_feat_size

    return args


class GCMCGraphConv(nn.Module, ABC):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))

        self.prob_score = nn.Linear(out_feats, 1, bias=False)
        self.review_score = nn.Linear(out_feats, 1, bias=False)
        self.review_w = nn.Linear(out_feats, out_feats, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.prob_score.weight)
        init.xavier_uniform_(self.review_score.weight)
        init.xavier_uniform_(self.review_w.weight)

    def forward(self, graph, feat, weight=None):

        with graph.local_scope():

            feat = self.weight

            graph.srcdata['h'] = feat
            review_feat = graph.edata['review_feat']

            graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))
            graph.edata['rf'] = self.review_w(review_feat) * torch.sigmoid(self.review_score(review_feat))
            graph.update_all(lambda edges: {'m': (edges.src['h'] * edges.data['pa'] + edges.data['rf'])
                                                 * self.dropout(edges.src['cj'])},
                             fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h'] * graph.dstdata['ci']

        return rst 


class GCMCLayer(nn.Module, ABC):

    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 out_units,
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(out_units, out_units)
        self.ifc = nn.Linear(out_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        sub_conv = {}
        for rating in rating_vals:
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             out_units,
                                             device=device,
                                             dropout_rate=dropout_rate)
            sub_conv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                 out_units,
                                                 device=device,
                                                 dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate='sum')
        self.agg_act = nn.GELU()
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {'user': ufeat, 'movie': ifeat}
        out_feats = self.conv(graph, in_feats)
        ufeat = out_feats['user']
        ifeat = out_feats['movie']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat


class ContrastLoss(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(th.Tensor(feat_size, feat_size))
        init.xavier_uniform_(self.w.data)
        #  self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x @ self.w * y ).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = th.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss

    def measure_sim(self, x, y):
        if len(y.shape) > len(x.shape):
            _l = y.shape[1]
            _x = x @ self.w
            _x = _x.unsqueeze(1)
            return (_x * y).sum(-1)

        else:
            return (x @ self.w * y_neg).sum(-1)


class MLPPredictorMI(nn.Module, ABC):
    """
    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """

    def __init__(self,
                 in_units,
                 num_classes,
                 dropout_rate=0.0,
                 neg_sample_size=1):
        super(MLPPredictorMI, self).__init__()
        self.neg_sample_size = neg_sample_size
        self.dropout = nn.Dropout(dropout_rate)

        self.contrast_loss = ContrastLoss(in_units)

        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.predictor = nn.Linear(in_units, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def neg_sampling(graph):
        review_feat = graph.edata['review_feat']
        neg_review_feat = review_feat[th.randperm(review_feat.shape[0]), :]
        return neg_review_feat

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h_fea = self.linear(th.cat([h_u, h_v], dim=1))
        score = self.predictor(h_fea).squeeze()

        if 'neg_review_feat' in edges.data:
            review_feat = edges.data['review_feat']
            # neg_review_feat = review_feat[th.randperm(review_feat.shape[0]), :]
            neg_review_feat = edges.data['neg_review_feat']
            mi_score = self.contrast_loss(h_fea, review_feat, neg_review_feat)
            return {'score': score, 'mi_score': mi_score}
        else:
            return {'score': score}

    def forward(self, graph, ufeat, ifeat, cal_edge_mi=True, neg_graph=False):
        graph.nodes['user'].data['h'] = ufeat
        graph.nodes['movie'].data['h'] = ifeat
        
        if not neg_graph:
            if ('review_feat' in graph.edata) & cal_edge_mi:
                graph.edata['neg_review_feat'] = self.neg_sampling(graph)
            else:
                del graph.edata['neg_review_feat']

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            if 'mi_score' in graph.edata:
                return graph.edata['score'], graph.edata['mi_score']
            else:
                return graph.edata['score']


class Net(nn.Module, ABC):
    def __init__(self, params):
        super(Net, self).__init__()
        self._params = params
        self.encoder = GCMCLayer(params.rating_vals,
                                 params.src_in_units,
                                 params.dst_in_units,
                                 params.gcn_out_units,
                                 dropout_rate=params.gcn_dropout,
                                 device=params.device)

        if params.train_classification:
            self.decoder = MLPPredictorMI(in_units=params.gcn_out_units,
                                        num_classes=len(params.rating_vals))
        else: 
            self.decoder = MLPPredictorMI(in_units=params.gcn_out_units,
                                        num_classes=1)
        # self.contrast_loss = ContrastLoss(params.gcn_out_units, x_inner=True)
        self.contrast_loss = ContrastLoss(params.gcn_out_units)

    def forward(self, enc_graph, dec_graph, neg_dec_graph, ufeat, ifeat, cal_edge_mi=True):
        #각 Embedding with Graph 정보
        user_out, movie_out = self.encoder(enc_graph, ufeat, ifeat)
        if self._params.distributed:
            user_out = user_out.to(self._params.device)
            movie_out = movie_out.to(self._params.device)

        if cal_edge_mi:
            pred_ratings, mi_score = self.decoder(dec_graph, user_out, movie_out, cal_edge_mi)
            neg_pred_ratings = self.decoder(neg_dec_graph, user_out, movie_out, False, True)
            return pred_ratings, neg_pred_ratings, mi_score, user_out, movie_out
        else:
            pred_ratings = self.decoder(dec_graph, user_out, movie_out, cal_edge_mi)
            neg_pred_ratings = self.decoder(neg_dec_graph, user_out, movie_out, False, True)

            return pred_ratings, neg_pred_ratings, user_out, movie_out

# def calculate_ndcg(relevance, k, num):
#     def dcg_at_k(r, k):
#         r = np.asfarray(r)[:k]
#         return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    
#     dcg = dcg_at_k(relevance, k)
#     idcg = dcg_at_k(np.ones(num), k)
#     return dcg / idcg if idcg > 0 else 0

# def evaluate(params, net, dataset, segment='valid'):
#     possible_rating_values = dataset.possible_rating_values

#     if segment == "valid":
#         rating_values = dataset.valid_truths
#         enc_graph = dataset.valid_enc_graph
#         dec_graph = dataset.valid_dec_graph
#         filters = np.concatenate((dataset.train_pair, dataset.test_pair), axis=1)
#         answer = dataset.valid_pair
#     elif segment == "test":
#         rating_values = dataset.test_truths
#         enc_graph = dataset.test_enc_graph
#         dec_graph = dataset.test_dec_graph
#         filters = np.concatenate((dataset.train_pair, dataset.valid_pair), axis=1)
#         answer = dataset.test_pair
#     else:
#         raise NotImplementedError

#     # Evaluate RMSE
#     net.eval()
#     user_num = dataset._num_user
#     user_num = 500 #경우의 수가 너무 많아 일단 500명으로 제한
#     item_num = dataset._num_movie

#     #(사용자와 아이템 조합의 모든 경우의 수) - (상호작용이 있는 사용자, 아이템 조합)
#     user = np.repeat(np.arange(user_num), item_num)
#     item = np.tile(np.arange(item_num), user_num)
#     filtered_set = set(zip(user, item)) - set(zip(filters[0], filters[1]))

#     filtered_user, filtered_item = zip(*filtered_set)

#     filtered_user = np.array(filtered_user, dtype=np.int64)
#     filtered_item = np.array(filtered_item, dtype=np.int64)
#     #튜플의 형태로 필터링된 결과 저장
#     filtered_result = (filtered_user, filtered_item)

#     filtered_dec_graph = dataset._generate_dec_graph(filtered_result).int().to(params.device)
#     k = 20 #상위 20개 추천

#     with th.no_grad():
#         pred_ratings, neg_rating, _, _, _ = net(enc_graph, dec_graph, filtered_dec_graph,
#                                     dataset.user_feature, dataset.movie_feature)
#         neg_rating = neg_rating.cpu().numpy()
#         ndcg = 0
#         recall = 0
#         hit_ratio = 0
#         normalize = user_num
#         for user in range(user_num):
#             indices = np.where(answer[0] == user)
#             interacted_item = answer[1][indices]
#             #상호작용 없는 인원 제외(정답 없음)
#             if len(interacted_item) == 0:
#                 normalize -= 1
#                 continue
#             indices = np.where(filtered_result[0] == user)
#             neg_item = (filtered_result[1][indices])
#             neg_item_rating = neg_rating[indices]
#             #내림차순 정렬된 인덱스 배열을 얻고 Top-k개 저장
#             sorted_indices = np.argsort(neg_item_rating)[::-1]
#             predicted_item = neg_item[sorted_indices][:k]
#             relevance = [1 if item in interacted_item else 0 for item in predicted_item]
#             nd = calculate_ndcg(relevance, k, len(interacted_item))
#             rec = sum(relevance)/len(interacted_item)
#             hit = min(sum(relevance), 1)
#             ndcg += nd
#             recall += rec
#             hit_ratio += hit
#     return ndcg/user_num, recall/user_num, hit_ratio/user_num

def calculate_ndcg(relevance, k, num):
    def dcg_at_k(r, k):
        result = 0
        r = np.asfarray(r)[:k]
        for i in range(len(r)):
            result += (np.power(2, k - r[i][1]) - 1) / np.log2(i+2)
        return result
    
    def idcg_at_k(k):
        result = 0
        for i in range(k):
            result += (np.power(2, k - i) - 1) / np.log2(i+2)
        return result
    
    dcg = dcg_at_k(relevance, k)
    idcg = idcg_at_k(k)
    return dcg / idcg if idcg > 0 else 0

def evaluate(params, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
        filters = np.concatenate((dataset.train_pair, dataset.test_pair), axis=1)
        answer = dataset.valid_pair
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
        filters = np.concatenate((dataset.train_pair, dataset.valid_pair), axis=1)
        answer = dataset.test_pair
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    user_num = dataset._num_user
    user_num = 500 #경우의 수가 너무 많아 일단 500명으로 제한
    item_num = dataset._num_movie

    #(사용자와 아이템 조합의 모든 경우의 수) - (상호작용이 있는 사용자, 아이템 조합)
    user = np.repeat(np.arange(user_num), item_num)
    item = np.tile(np.arange(item_num), user_num)
    filtered_set = set(zip(user, item)) - set(zip(filters[0], filters[1]))

    filtered_user, filtered_item = zip(*filtered_set)

    filtered_user = np.array(filtered_user, dtype=np.int64)
    filtered_item = np.array(filtered_item, dtype=np.int64)
    #튜플의 형태로 필터링된 결과 저장
    filtered_result = (filtered_user, filtered_item)

    filtered_dec_graph = dataset._generate_dec_graph(filtered_result).int().to(params.device)
    k = 20 #Top-k개 추천

    with th.no_grad():
        pred_ratings, neg_rating, _, _, _ = net(enc_graph, dec_graph, filtered_dec_graph,
                                    dataset.user_feature, dataset.movie_feature)
        neg_rating = neg_rating.cpu().numpy()
        ndcg = 0
        recall = 0
        hit_ratio = 0
        normalize = user_num
        for user in range(user_num):
            indices = np.where(answer[0] == user)
            interacted_item = answer[1][indices]
            interacted_item_rating = rating_values[indices]
            sorted_indices = np.argsort(interacted_item_rating.cpu().numpy())[::-1]
            answer_item_ranking = (np.array(interacted_item[sorted_indices]), np.arange(len(interacted_item)))
            #상호작용 없는 인원 제외(정답 없음)
            if len(interacted_item) == 0:
                normalize -= 1
                continue
            indices = np.where(filtered_result[0] == user)
            neg_item = (filtered_result[1][indices])
            neg_item_rating = neg_rating[indices]
            #내림차순 정렬된 인덱스 배열을 얻고 Top-k개 저장
            sorted_indices = np.argsort(neg_item_rating)[::-1]
            predicted_item = neg_item[sorted_indices][:k]

            relevance = []
            for item in predicted_item:
                if item in answer_item_ranking[0]:
                    index = np.where(answer_item_ranking[0] == item)
                    relevance.append((answer_item_ranking[0][index], answer_item_ranking[1][index]))
                #else 없는 경우 처리해주어야 하는데 까먹음. 수정 요망


            nd = calculate_ndcg(relevance, k, len(interacted_item))
            rec = len(relevance)/len(interacted_item)
            hit = min(len(relevance), 1)
            ndcg += nd
            recall += rec
            hit_ratio += hit
    return ndcg/user_num, recall/user_num, hit_ratio/user_num

def positive_pairs(users, items):
    positive_pairs = {}
    for user, item in zip(users,items):
        if user not in positive_pairs:
            positive_pairs[user] = set()
        positive_pairs[user].add(item)
    return positive_pairs

def sample_negative(positive_pairs, users, items, device='cuda'):
    unique_items = th.tensor(np.unique(items), device=device)
    
    negative_pairs = []
    
    negative_item_cache = {user: set() for user in np.unique(users)}

    for user, item in zip(users, items):
        positive_items = th.tensor(list(positive_pairs[user]), device=device)
        
        non_interacted_items = unique_items[~th.isin(unique_items, positive_items)]
        
        available_items = [x for x in non_interacted_items.tolist() if x not in negative_item_cache[user]]
        
        if available_items:
            neg_item = random.choice(available_items)
            negative_pairs.append((user, neg_item))
            negative_item_cache[user].add(neg_item) 
    return np.array(negative_pairs)

def sample_neg_pair(A,B):
    A=A
    neg_B = B[th.randperm(B.shape[0])]    
    return (np.array(A, dtype=np.int64), np.array(neg_B, dtype=np.int64))

def train(params):
    # wandb.init(name=params.model_short_name,
    #            config=params,
    #            project=params.dataset_name,
    #            config_include_keys=['review_feat_size',
    #                                 'gcn_dropout',
    #                                 'ed_alpha',
    #                                 'nd_alpha',
    #                                 'model_save_path'],
    #            mode='offline',
    #            save_code=True)
    print(params)
    dataset = MovieLens(params.dataset_name,
                        params.dataset_path,
                        params.device,
                        params.review_feat_size,
                        symm=True)
    print("Loading data finished ...\n")

    params.src_in_units = dataset.user_feature_shape[1]
    params.dst_in_units = dataset.movie_feature_shape[1]
    params.rating_vals = dataset.possible_rating_values

    net = Net(params)

    if params.distributed:
        net.encoder.cpu()
        net.decoder.to(params.device)
        net.contrast_loss.to(params.device)
        nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(params.device)
    else:
        net = net.to(params.device)
        nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(params.device)

    rating_loss_net = nn.CrossEntropyLoss() if params.train_classification else nn.MSELoss()
    learning_rate = params.train_lr
    # optimizer = get_optimizer(params.train_optimizer)(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    # prepare training data
    if params.train_classification:
        train_gt_labels = dataset.train_labels
        train_gt_ratings = dataset.train_truths
    else:
        train_gt_labels = dataset.train_truths.float()
        train_gt_ratings = dataset.train_truths.float()

    if params.distributed:
        train_gt_labels = train_gt_labels.to(params.device)
        train_gt_ratings = train_gt_ratings.to(params.device)

    # declare the loss information
    best_valid_rmse = np.inf
    best_test_rmse = np.inf
    no_better_valid = 0
    best_iter = -1

    if params.distributed:
        dataset.train_enc_graph = dataset.train_enc_graph.int().cpu()
        dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
        dataset.valid_enc_graph = dataset.train_enc_graph
        dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
        dataset.test_enc_graph = dataset.test_enc_graph.int().cpu()
        dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    else:
        dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)
        dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
        dataset.valid_enc_graph = dataset.train_enc_graph
        dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
        dataset.test_enc_graph = dataset.test_enc_graph.int().to(params.device)
        dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)


    #pp = positive_pairs(dataset.train_pair[0], dataset.train_pair[1])
    #train_neg_pairs = sample_negative(pp, dataset.train_pair[0], dataset.train_pair[1]).T
    #neg_dec_graph = dataset._generate_dec_graph(train_neg_pairs).int().to(params.device)
    print("Start training ...")
    ndcg = recall = hit_ratio = 0
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        optimizer.zero_grad()

        # 상호작용이 없는 사용자-아이템 쌍 리턴 
        train_neg_pairs = sample_neg_pair(dataset.train_pair[0], dataset.train_pair[1])
        # 상호작용 없는 관계끼리 연결된 새로운 그래프 리턴
        neg_dec_graph = dataset._generate_dec_graph(train_neg_pairs).int().to(params.device)

        # pos_prediction, 2 : 실제 평가가 이루어진 아이템들에 대한 예측 평점 리스트
        # neg_prediction, 2 : 평가가 이루어진 적 없는 아이템들에 대한 예측 평점 리스트
        pos_prediction, neg_prediction, ed_mi1, user1, item1 = net(dataset.train_enc_graph,
                                                  dataset.train_dec_graph,
                                                  neg_dec_graph,
                                                  dataset.user_feature,
                                                  dataset.movie_feature,)
        _, _, ed_mi2, user2, item2 = net(dataset.train_enc_graph,
                                                  dataset.train_dec_graph,
                                                  neg_dec_graph,
                                                  dataset.user_feature,
                                                  dataset.movie_feature)
        user_mi_loss = net.contrast_loss(user1, user2).mean()
        item_mi_loss = net.contrast_loss(item1, item2).mean()

        nd_loss = (user_mi_loss + item_mi_loss) / 2
        # nd_loss = item_mi_loss
        ed_loss = (ed_mi1.mean() + ed_mi2.mean()) / 2

        # BPR Loss 계산
        bpr_loss = -(pos_prediction - neg_prediction).sigmoid().log().mean()

        # 기존 RMSE를 활용한 Loss
        # total_loss = r_loss + params.nd_alpha * nd_loss + params.ed_alpha * ed_loss
        # BPR Total Loss
        total_loss = bpr_loss + params.nd_alpha * nd_loss + params.ed_alpha * ed_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
        optimizer.step()

        if params.train_classification:
            real_pred_ratings = (th.softmax(pos_prediction, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        else: 
            real_pred_ratings = pos_prediction

        train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean().sqrt()
        if iter_idx % 10 == 0:
            ndcg, recall, hit_ratio = evaluate(params=params, net=net, dataset=dataset, segment='valid')
            logging_str = f"Iter={iter_idx:>4d}, Train_RMSE={train_rmse:.4f}, ED_MI={ed_loss:.4f}, ND_MI={nd_loss:.4f}, " \
                      f"NDCG={ndcg.item():.4f}, Recall={recall:.4f}, Hit_ratio={hit_ratio:.4f}, Train_BPR_Loss={bpr_loss:.4f},"
            print(logging_str)
        if  iter_idx in [200, 300, 400]:
            new_lr = learning_rate * params.train_lr_decay_factor
            for p in optimizer.param_groups:
                p['lr'] = new_lr
            print(new_lr)
            

    #print(f'Best Iter Idx={best_iter:>4d}, Best Valid RMSE={best_valid_rmse:.4f}, Best Test RMSE={best_test_rmse:.4f}')
    print(params.model_save_path)


if __name__ == '__main__':
    config_args = config()

    train(config_args)
