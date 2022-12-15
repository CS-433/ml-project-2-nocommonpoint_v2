import numpy as np
import matplotlib.pyplot as plt

def per_class_accuracy(y, y_pred):
    min_label = y.min()
    max_label = y.max()
    recalls = []
    for l in range(min_label, max_label + 1):
        in_class = y == l
        tp = (in_class & (y_pred == l)).sum()
        fn = (in_class & (y_pred != l)).sum()
        rec = tp / (tp + fn)
        recalls.append(rec)
    return np.array(recalls)

def per_class_corrects_totals(y, y_pred):
    min_label = y.min()
    max_label = y.max()
    corrects, totals = [], []
    for l in range(min_label, max_label + 1):
        in_class = y == l
        tp = (in_class & (y_pred == l)).sum()
        total = in_class.sum()
        corrects.append(tp)
        totals.append(total)
    return np.array(corrects), np.array(totals)

def accuracy_info(y, y_pred, plot=True, prefix='', ax=None, verbose=True):
    y = y.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    mean_acc = (y == y_pred).mean()
    per_class_acc = per_class_accuracy(y, y_pred)
    balanced_acc = per_class_acc[np.isfinite(per_class_acc)].mean()
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.bar(np.arange(y.min(), y.max() + 1), per_class_acc)
        ax.set_title(f'{prefix} Per-Class Recalls')
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(0.00, 1.01, 0.05))
        ax.grid()
    if verbose:
        print(f'Mean accuracy: {100 * mean_acc:.3f}%')
        print(f'Balanced accuracy: {100 * balanced_acc:.3f}%')
    return balanced_acc, mean_acc

def correct_count_info(y, y_pred):
    y = y.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    return per_class_corrects_totals(y, y_pred)

