# -*- coding: utf-8 -*-

"""
Pytorch port of Relation Knowledge Distillation Losses.

credits:
    https://github.com/lenscloth/RKD/blob/master/metric/utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSE(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019."""
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, fea_old, fea_new):
        alpha=torch.rand(1)
        f = nn.MSELoss(reduction='mean')
        fea_new_c = self.channel_similarity(fea_new)
        fea_old_c = self.channel_similarity(fea_old)

        fea_new_mean = self.get_fea_mean(fea_new)
        fea_old_mean = self.get_fea_mean(fea_old)

        loss = alpha * f(fea_old_mean, fea_new_mean) + (1-alpha) * f(fea_old_c, fea_new_c)

        return loss

    def channel_similarity(self, fm):  # channel_similarity
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
        s = norm_fm.bmm(norm_fm.transpose(1, 2))
        s = s.unsqueeze(1)
        return s

    def get_fea_mean(self, feature):
        size = feature.size()
        N, C = size[:2]
        feat_mean = feature.view(N, C, -1).mean(dim=2)
        feature = torch.squeeze(feat_mean)  # 得到feature[128,64]
        return feat_mean

#自定义ce loss
class MSE(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019."""
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, feas_new, feas_old):
        v=0.0
        f = nn.MSELoss(reduction='mean')
        for i in range(len(feas_old)):
            fea_new = feas_new[i]
            fea_old = feats_old[i]
            fea_new = self.channel_similarity(fea_new)
            fea_old = self.channel_similarity(fea_old)
            v+=f(fea_new, fea_old)
        loss = v.mean()
        return loss

    def channel_similarity(self, fm):  # channel_similarity
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
        s = norm_fm.bmm(norm_fm.transpose(1, 2))
        s = s.unsqueeze(1)
        return s



# circle Loss
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

class BoundaryLoss(nn.Module):

    def __init__(self): #num_labels逐步增加[10,20,...90]  num_labels=10, feat_dim=64

        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        num_labels = 100
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        nn.init.normal_(self.delta)

    # 传进来的delta是确定的
    def forward(self, pooled_output, centroids, labels):
        if pooled_output == None:
            loss = 0.0
        else:
            # delta_ = torch.randn(delta.shape)
            # delta_.copy_(delta)
            # delta_ = delta_.cuda()
            # nn.init.normal_(delta_)
            # logits = euclidean_metric(pooled_output, centroids)
            # probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            # print('logits:', logits.shape, logits, probs, preds) 计算属于哪类
            delta = F.softplus(delta)  # +le-5
            c = centroids[labels]  # [128,64]
            d = delta[labels]
            x = pooled_output  # [128,64]

            euc_dis = torch.norm(x - c, 2, 1).view(-1)  # [128]
            # print('euc:', euc_dis)
            pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
            neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
            #print('mask:', pos_mask)
            pos_loss = (euc_dis - d) * pos_mask
            neg_loss = (d - euc_dis) * neg_mask
            loss = pos_loss.mean() + neg_loss.mean()

        return loss


class BoundaryLoss(nn.Module):

    def __init__(self):  # , num_labels=10, feat_dim=64): #num_labels逐步增加[10,20,...90]

        super(BoundaryLoss, self).__init__()
        # self.num_labels = num_labels
        # self.feat_dim = feat_dim
        # self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        # nn.init.normal_(self.delta)

    # 传进来的delta是确定的
    def forward(self, pooled_output, centroids, delta, labels):
        if pooled_output == None:
            loss = 0.0
        else:
            # delta_ = torch.randn(delta.shape)
            # delta_.copy_(delta)
            # delta_ = delta_.cuda()
            # nn.init.normal_(delta_)
            # logits = euclidean_metric(pooled_output, centroids)
            # probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            # print('logits:', logits.shape, logits, probs, preds) 计算属于哪类
            #delta_ = F.softplus(delta_)  # +le-5
            c = centroids[labels]  # [128,64]
            d = delta[labels]
            x = pooled_output  # [128,64]

            euc_dis = torch.norm(x - c, 2, 1).view(-1)  # [128]
            # print('euc:', euc_dis)
            #pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
            #neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
            # print('mask:', pos_mask)
            # pos_loss = (euc_dis - d) * pos_mask
            # neg_loss = (d - euc_dis) * neg_mask
            # loss = pos_loss.mean() + neg_loss.mean()
            neg_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
            loss = ((euc_dis - d) * neg_mask).mean()

        return loss
#######
class BoundaryLoss(nn.Module):

    def __init__(self):

        super(BoundaryLoss, self).__init__()

    def forward(self, fea_old, fea_new, centroids, labels):

        c = centroids[labels]  # [128,64]

        old_dis = torch.norm(fea_old - c, 2, 1).view(-1)  # [128]
        new_dis = torch.norm(fea_new - c, 2, 1).view(-1)  # [128]
        loss = nn.MSELoss(reduction='mean')(new_dis, old_dis)+nn.MSELoss(reduction='mean')(fea_new, fea_old)

        return loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(
        min=eps
    )

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKDAngleLoss(nn.Module):
    def __init__(
            self,
            in_dim1: int = 0,
            in_dim2: int = None,
            proj_dim: int = None,
    ):
        super().__init__()

        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim1 if in_dim2 is None else in_dim2

        if proj_dim is None:
            proj_dim = min(self.in_dim1, self.in_dim2)

        self.proj_dim = proj_dim

        self.embed1 = self.embed2 = nn.Identity()
        if in_dim1 > 0:
            self.embed1 = nn.Linear(self.in_dim1, self.proj_dim)
            self.embed2 = nn.Linear(self.in_dim2, self.proj_dim)

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        student, teacher = self.embed1(student), self.embed2(teacher)

        td = teacher.unsqueeze(0) - teacher.unsqueeze(1)  # ta-tb
        norm_td = F.normalize(td, p=2, dim=2)  # l2
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)  # cos值

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.l1_loss(s_angle, t_angle)
        return loss

class KDLoss(nn.Module):
    def __init__(
            self,
            in_dim1: int = 0,
            in_dim2: int = None,
            proj_dim: int = None,
    ):
        super().__init__()

        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim1 if in_dim2 is None else in_dim2

        if proj_dim is None:
            proj_dim = min(self.in_dim1, self.in_dim2)

        self.proj_dim = proj_dim

        self.embed1 = self.embed2 = nn.Identity()
        if in_dim1 > 0:
            self.embed1 = nn.Linear(self.in_dim1, self.proj_dim)
            self.embed2 = nn.Linear(self.in_dim2, self.proj_dim)

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        loss = 0.0
        for i in range(len(student)):
            student[i], teacher[i] = self.embed1(student[i]), self.embed2(teacher[i])
            loss += nn.MSELoss()(student[i], teacher[i])
        loss = loss/2
        return loss

class SM(nn.Module):
    def __init__(self):
        super(SM, self).__init__()

    def forward(self, fea_stu, fea_tea):
        # m_s = torch.mean(fea_stu, [1])
        # m_t = torch.mean(fea_tea, [1])
        # v_s = torch.var(fea_stu, [1])
        # v_t = torch.var(fea_tea, [1])
        m_s, v_s = self.calc_mean_std(fea_stu)
        m_t, v_t = self.calc_mean_std(fea_tea)
        loss = (nn.MSELoss()(m_s, m_t)+nn.MSELoss()(v_s, v_t)).mean() #nn.L1Loss()
        return loss

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


class RKDDistanceLoss(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d)
        return loss


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1., tau=4.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)

        y_s = (y_s/self.tau).softmax(dim=1)
        y_t = (y_t/self.tau).softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)   #类间
        intra_loss = intra_class_relation(y_s, y_t)   #类内
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss

class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019."""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

class SingleDIST(nn.Module):
    def __init__(self, beta=1., gamma=1., tau=4.0):
        super(SingleDIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)

        y_s = (y_s/self.tau).softmax(dim=1)
        y_t = (y_t/self.tau).softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)   #类间
        #intra_loss = intra_class_relation(y_s, y_t)   #类内
        loss = self.beta * inter_loss #+ self.gamma * intra_loss
        return loss

class SM(nn.Module):
    def __init__(self):
        super(SM, self).__init__()

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, fea_stu, fea_tea):
        m_s, v_s = self.calc_mean_std(fea_stu)
        m_t, v_t = self.calc_mean_std(fea_tea)
        #loss = (nn.MSELoss()(m_s, m_t)+nn.MSELoss()(v_s, v_t)).mean()
        loss = -cosine_similarity(m_s,m_t).mean()-cosine_similarity(v_s,v_t).mean()
        return loss

#添加约束
class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, y, yh):
        p = torch.nn.functional.softmax(yh, dim=1)
        log_p = torch.nn.functional.log_softmax(yh, dim=1)
        loss = - (y * log_p).sum(dim=1).mean()
        return loss

class DistLoss_(nn.Module):
    def __init__(
            self,
            in_dim1: int = 0,
            in_dim2: int = None,
            proj_dim: int = None,
    ):
        super().__init__()

        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim1 if in_dim2 is None else in_dim2

        if proj_dim is None:
            proj_dim = min(self.in_dim1, self.in_dim2)

        self.proj_dim = proj_dim

        self.embed1 = self.embed2 = nn.Identity()
        if in_dim1 > 0:
            self.embed1 = nn.Linear(self.in_dim1, self.proj_dim)
            self.embed2 = nn.Linear(self.in_dim2, self.proj_dim)
        self.beta = 1.0
        self.gamma = 1.0
        self.tau = 4.0

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        # N x C
        # y_s, y_t = self.embed1(y_s), self.embed2(y_t)
        #N C W H -> N W*H
        y_s = torch.mean(y_s, axis=1)
        y_s = y_s.view(y_s.size(0), -1)
        y_t = torch.mean(y_t, axis=1)
        y_t = y_t.view(y_t.size(0), -1)

        y_s, y_t = self.embed1(y_s), self.embed2(y_t)

        y_s = (y_s / self.tau).softmax(dim=1)
        y_t = (y_t / self.tau).softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        #loss = nn.CosineEmbeddingLoss()(y_s, y_t, torch.ones())
        # y_s = spatial_similarity(y_s)
        # y_t = spatial_similarity(y_t)
        # loss = nn.MSELoss(reduction='mean')(y_s, y_t)
        return loss

class PredictLoss(nn.Module):
    def __init__(
            self,
            in_dim: int = 0,
            out_dim: int = None,
            proj_dim: int = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        if proj_dim is None:
            proj_dim = min(self.in_dim, self.out_dim)

        self.proj_dim = proj_dim

        self.layer1 = self.layer2 = nn.Identity()
        if in_dim1 > 0:
            self.layer1 = nn.Sequential(
                nn.Linear(self.in_dim, self.proj_dim),
                nn.BatchNorm1d(self.proj_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = nn.Linear(self.proj_dim, self.out_dim)

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        y_s = self.layer1(y_s)
        y_s = self.layer2(y_s)
        loss = nn.MSELoss()(y_s, y_t).mean()

        return loss

class DISTOLD(nn.Module):
    def __init__(self, beta=1., gamma=2.):
        super(DISTOLD, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        # if y_s.ndim == 4:
        #     num_classes = y_s.shape[1]
        #     y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
        #     y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss

class DISTNEW(nn.Module):
    def __init__(self, beta=2., gamma=1.):
        super(DISTNEW, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss

def batch_similarity(fm):  # batch similarity
        fm = fm.view(fm.size(0), -1)
        Q = torch.mm(fm, fm.transpose(0, 1))
        normalized_Q = Q / torch.norm(Q, 2, dim=1).unsqueeze(1).expand(Q.shape)
        return normalized_Q

def spatial_similarity(fm):  # spatial similarity
    fm = fm.view(fm.size(0), fm.size(1), -1)  #[b,c,wh]
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    s = norm_fm.transpose(1, 2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm):  # channel_similarity
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
        s = norm_fm.bmm(norm_fm.transpose(1, 2))
        s = s.unsqueeze(1)
        return s



# class ATTN(nn.Module):
#     def __init__(self):
#         super(ATTN, self).__init__()
#
#     # def forward(self, attention_map1, attention_map2):
#     # """Calculates the attention distillation loss"""
#     # attention_map1 = torch.norm(attention_map1, p=2, dim=1)
#     # attention_map2 = torch.norm(attention_map2, p=2, dim=1)
#     # return torch.norm(attention_map2 - attention_map1, p=1, dim=1).sum(dim=1).mean()
#     def forward(self, feature_o, out_o, feature_n, out_n):
#         batch = out_n.size()[0]
#         index = out_n.argmax(dim=-1).view(-1, 1)
#         onehot = torch.zeros_like(out_n)
#         onehot.scatter_(-1, index, 1.)
#         out_o, out_n = torch.sum(onehot * out_o), torch.sum(onehot * out_n)
#         out_o.requires_grad_(True)
#         feature_o.requires_grad_(True)
#         #print('grad:',out_o.requires_grad, feature_o.requires_grad)
#         grads_o = torch.autograd.grad(out_o, feature_o)[0]
#         grads_n = torch.autograd.grad(out_n, feature_n, create_graph=True)[0]
#         weight_o = grads_o.mean(dim=(2, 3)).view(batch, -1, 1, 1)
#         weight_n = grads_n.mean(dim=(2, 3)).view(batch, -1, 1, 1)
#         out_o.requires_grad_(False)
#         feature_o.requires_grad_(False)
#         cam_o = F.relu((grads_o * weight_o).sum(dim=1))
#         cam_n = F.relu((grads_n * weight_n).sum(dim=1))
#
#         # normalization
#         cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
#         cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)
#
#         loss_AD = (cam_o - cam_n).norm(p=1, dim=1).mean()
#
#         return loss_AD
#
#
# class CONS(nn.Module):
#     def __init__(self):
#         super(CONS, self).__init__()
#
#     # def forward(self, fea_new, fea_old, m):
#     #     return nn.CosineEmbeddingLoss()(fea_new, fea_old, m)
#
#
def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        # (64,64,32,32)->(64,64,32,32)->(64,32,32)->(64,1024)->(64,1024)
        return 1-F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))  # 沿通道取mean

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    # (64,1024)-(64,1024)->(64,1024)->()
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


class AT(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    src code: https://github.com/szagoruyko/attention-transfer
    """
    def __init__(self):
        super(AT, self).__init__()

    def forward(self, g_s, g_t):
        p=2  #2
        return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])














