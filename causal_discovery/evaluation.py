import igraph as ig
import numpy as np


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            #raise ValueError('B_est should be a DAG')
            print('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = round(float(len(reverse) + len(false_pos)) / max(pred_size, 1), 4)
    tpr = round(float(len(true_pos)) / max(len(cond), 1), 4)
    fpr = round(float(len(reverse) + len(false_pos)) /
                max(cond_neg_size, 1), 4)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    # shd = len(extra_lower) + len(missing_lower) + len(reverse)
    extra = len(extra_lower)
    missing = len(missing_lower)
    reverse = len(reverse)
    shd = extra + missing + reverse
    precision = round(float(len(true_pos)) / max(len(pred), 1), 4)
    recall = round(float(len(true_pos)) / max(len(cond), 1), 4)
    f1 = round(2 * precision * recall / max(precision + recall, 1e-8), 4)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, "precision": precision, "recall": recall, "f1": f1,
            'extra': extra, 'missing': missing,  'reverse': reverse, 'shd': shd, 'nnz': pred_size}

def numerical_SHD(B_true, B_est):
    numerical_SHD_noextra=sum(sum(abs(B_true- np.where(B_true == 0, 0, B_est))))
    numerical_SHD_noextra=round(numerical_SHD_noextra,4)
    numerical_SHD=sum(sum(abs(B_true-B_est)))
    numerical_SHD=round(numerical_SHD,4)
    return {'numerical_SHD_noextra':numerical_SHD_noextra,'numerical_SHD':numerical_SHD}

def sid(tar,pred):
    try:
        from cdt.metrics import SID
        return {'SID':SID(tar, pred).item()}
    except:
        return {'SID':None}