import torch
import numpy as np
import wandb

def mpa(preds, labels):
    preds = torch.cat(preds).argmax(dim=1).reshape(-1)
    labels = torch.cat(labels).argmax(dim=1).reshape(-1)

    correct_pixels = torch.sum(preds == labels)
    total_pixels = len(preds)
    pixel_acc = correct_pixels / total_pixels / 3  # 3 channels (RGB)

    return pixel_acc


def iou(preds, labels):
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    
    num_examples, num_classes, _, _ = preds.shape

    # Reshape predictions and labels for calculations
    preds_flat = preds.view(num_examples * num_classes, -1)
    labels_flat = labels.view(num_examples * num_classes, -1)

    # Calculate intersection and union for each class
    intersection = np.sum(np.logical_and(preds_flat.cpu().numpy(), labels_flat.cpu().numpy()), axis=1)
    union = np.sum(np.logical_or(preds_flat.cpu().numpy(), labels_flat.cpu().numpy()), axis=1)

    # Reshape back to per-class values
    intersection_per_class = intersection.reshape(num_examples, num_classes)
    union_per_class = union.reshape(num_examples, num_classes)

    # Calculate mIoU
    miou = np.mean(intersection_per_class / (union_per_class + 1e-10), axis=1)  # Adding a small epsilon to avoid division by zero

    return np.mean(miou)


def fwiou(preds, labels):
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    
    num_classes = preds.shape[1]

    # Create one-hot encodings for predictions and labels
    preds_one_hot = np.eye(num_classes)[preds.argmax(dim=1).cpu().numpy()]
    labels_one_hot = np.eye(num_classes)[labels.argmax(dim=1).cpu().numpy()]

    # Calculate true positives, true positives + false negatives, and false positives + true negatives
    tp = np.sum(preds_one_hot * labels_one_hot, axis=(0, 2, 3))
    tp_fn = np.sum(labels_one_hot, axis=(0, 2, 3))
    tp_fp = np.sum(preds_one_hot, axis=(0, 2, 3))

    # Calculate frequency weighted IoU
    f_iou = np.sum(tp) / (np.sum(tp_fn) + np.sum(tp_fp) - np.sum(tp))

    return f_iou


def model_predict(model, test_dataloader):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input, label = batch
            pred = model(input)
            preds.append(pred)
            labels.append(label)

    preds = [pred.cpu() for pred in preds]
    labels = [label.cpu() for label in labels]

    return preds, labels


def model_eval(model, test_dataloader):
    preds, labels = model_predict(model, test_dataloader)

    mpa_value = mpa(preds, labels)
    iou_value = iou(preds, labels)
    fwiou_value = fwiou(preds, labels)

    wandb.summary['mpa'] = mpa_value
    wandb.summary['iou'] = iou_value
    wandb.summary['fwiou'] = fwiou_value

    print('mpa = {}, iou = {}, fwiou = {}'.format(mpa_value, iou_value, fwiou_value))
