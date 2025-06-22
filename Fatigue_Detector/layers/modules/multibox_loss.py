import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """RetinaFace使用的多任务损失函数
    
    实现功能：
    1. 计算三种损失：
       - 人脸分类损失(Confidence Loss)：交叉熵损失，区分人脸和背景
       - 人脸框回归损失(Localization Loss)：Smooth L1损失，精调人脸框位置
       - 人脸关键点损失(Landmark Loss)：Smooth L1损失，精调5个关键点位置
    2. 难例挖掘(Hard Negative Mining)：
       - 自动选择难以分类的负样本进行训练
       - 控制负样本和正样本的比例(默认3:1)
    
    数学公式：
    L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    其中：
    - Lconf: 分类损失(CrossEntropy)
    - Lloc: 定位损失(Smooth L1)
    - α: 平衡权重(默认为1)
    - N: 匹配的先验框数量
    
    输入输出：
    - 输入: 网络预测值、先验框、真实标注
    - 输出: 三个损失值组成的元组(定位损失, 分类损失, 关键点损失)
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """执行前向损失计算
        
        参数说明：
        predictions: 网络预测结果，包含三个元素的元组：
            - loc_preds: 人脸框偏移预测 [batch_size, num_priors, 4]
            - conf_preds: 人脸分类预测 [batch_size, num_priors, num_classes]
            - landm_preds: 关键点预测 [batch_size, num_priors, 10]
        priors: 先验框坐标 [num_priors, 4]
        targets: 真实标注 [batch_size, num_objs, 15] 
            - 前4维: 人脸框坐标(x1,y1,x2,y2)
            - 中间10维: 5个关键点坐标(x1,y1,...,x5,y5)
            - 最后1维: 类别标签(1:人脸, 0:背景)
            
        处理流程：
        1. 数据准备：解包预测结果，初始化目标张量
        2. 样本匹配：将先验框与真实标注进行匹配
        3. 损失计算：
           - 关键点损失：计算正样本的关键点坐标误差
           - 框回归损失：计算正样本的框位置误差
           - 分类损失：计算正负样本的分类误差
        4. 结果返回：归一化后的三种损失值
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # 1. 数据准备阶段：解包预测结果
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # 2. 关键点损失计算(仅正样本)
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != zeros
        conf_t[pos] = 1

        # 3. 框回归损失计算(仅正样本)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # 4. 分类损失计算(含难例挖掘)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 4.1 难例挖掘处理
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 4.2 计算最终分类损失
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # 5. 损失值归一化处理
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
