import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count

        # freq = cluster_sizes.float() / V
        # tau = cfg.model.get('tau', 1.5)
        # inv_freq = 1.0 / (freq + 1e-8)  # avoid division by zero
        # weight = inv_freq ** (1.0/ tau)  # inverse frequency raised to the power of tau
        # weight = weight / weight.sum()  # normalize weights to sum to 1
        # import pdb; pdb.set_trace()
        weight = (V - cluster_sizes).float() / V
        # import pdb; pdb.set_trace()
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)
        
def focal_loss(pred, true):
    """Focal loss for binary classification with optional class balancing.
    """
    if cfg.model.loss_fun == 'focal_loss':
        V = true.size(0)
        assert pred.ndim == 1
        n_classes = pred.shape[1] if pred.ndim > 1 else 2

        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()  # avoid NaN
        
        alpha = weight[1]  # assuming class 1 is the rare class
        max_gamma = 2.0 # cfg.model.get('focal_gamma', 2.0)
        gamma = max_gamma * (cfg.optim.current_epoch / max(1, cfg.optim.max_epoch))  # linearly increase gamma

        # Compute focal loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, true.float(), reduction='none')
        probas = torch.sigmoid(pred)
        pt = probas * true + (1 - probas) * (1 - true)  # p_t
        focal_weight = (1 - pt) ** gamma
        loss = alpha * focal_weight * bce_loss
        return loss.mean(), probas

def focal_with_dice(pred, true):
    if cfg.model.loss_fun == 'focal_with_dice':
        V = true.size(0)
        assert pred.ndim == 1
        n_classes = pred.shape[1] if pred.ndim > 1 else 2

        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()  # avoid NaN
        
        alpha = weight[1]  # assuming class 1 is the rare class
        max_gamma = 3.0 # cfg.model.get('focal_gamma', 2.0)
        gamma = max_gamma * (cfg.optim.current_epoch / max(1, cfg.optim.max_epoch))  # linearly increase gamma

        # Compute focal loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, true.float(), reduction='none')
        pred_prob = torch.sigmoid(pred)
        pt = pred_prob * true + (1 - pred_prob) * (1 - true)  # p_t
        focal_weight = (1 - pt) ** gamma
        loss = alpha * focal_weight * bce_loss
        eps = 1e-8
        intersection = (pred_prob * true.float()).sum()
        dice_loss = 1 - (2. * intersection + eps) / (pred_prob.sum() + true.float().sum() + eps)
        loss = loss + 0.1 * dice_loss

        return loss.mean(), pred_prob

def weighted_cross_entropy_with_dice(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy_with_dice':
        assert pred.ndim == 1
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                    weight=weight[true])
        pred_prob = torch.sigmoid(pred)
        # calculate dice loss
        eps = 1e-8
        intersection = (pred_prob * true.float()).sum()
        dice_loss = 1 - (2. * intersection + eps) / (pred_prob.sum() + true.float().sum() + eps)
        loss = loss + 0.5 * dice_loss
        return loss, pred_prob


register_loss('weighted_cross_entropy', weighted_cross_entropy)

register_loss('weighted_cross_entropy_with_dice', weighted_cross_entropy_with_dice)

register_loss('focal_loss', focal_loss)

register_loss('focal_wigh_dice', focal_with_dice)