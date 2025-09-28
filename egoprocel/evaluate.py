import cv2
import numpy as np

import pandas as pd

import torch

from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

def _sample_clip(frames, num_frames_to_sample):
    num_frames = len(frames)
    error_message = 'Can\'t sample more frames than there are in the video'
    assert num_frames >= num_frames_to_sample, error_message
    lower_lim = np.floor(num_frames/num_frames_to_sample)
    upper_lim = np.ceil(num_frames/num_frames_to_sample)
    count = np.arange(1, frames.shape[0] + 1)
    lower_mask = (count % lower_lim) == 0
    lower_frames = frames[lower_mask]
    upper_mask = (count % upper_lim) == 0
    upper_frames = frames[upper_mask]
    if len(upper_frames) < num_frames_to_sample:
        return (
            lower_frames,
            lower_mask * np.ones(lower_mask.shape, dtype=np.int8)
        )
    else:
        return (
            upper_frames,
            upper_mask * np.ones(upper_mask.shape, dtype=np.int8)
        )


def sample_labels(dset_name, video_path, annotation_path):
    videocap = cv2.VideoCapture(video_path)
    fps = int(videocap.get(cv2.CAP_PROP_FPS))
    length = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
    videocap.release()
    annotation_data = pd.read_csv(
        open(annotation_path, 'r'),
        header=None,
        encoding='utf8'
    )
    # print(video_path, fps)
    frames = np.arange(0, length)
    video_duration = len(frames) / fps
    sampling_fps = fps / 2.0
    candidate_frames, mask = _sample_clip(frames, int(video_duration * sampling_fps))
    
    labels_ = gen_labels(fps, annotation_data.values, length, dataset_name=dset_name)
    labels_mask = labels_ * mask
    labels = list()
    for label in labels_mask:
        if label != 0:
            if label == -1:
                labels.append(0)
            else:
                labels.append(label)
    return candidate_frames, mask, np.array(labels)


def gen_labels(fps, annotation_data, num_frames, dataset_name=None):
    """
    This method is used to generate labels for the test dataset.

    Args:
        fps (int): frame per second of the video
        annotation_data (list): list of procedure steps
        num_frames (int): number of frames in the video

    Returns:
        labels (ndarray): numpy array of labels with length equal to the
            number of frames
    """
    labels = np.ones(num_frames, dtype=int)*-1
    for step in annotation_data:
        if dataset_name == 'CrossTask':
            start_time = step[1]
            end_time = step[2]
            label = step[0]
        else:
            start_time = step[0]
            end_time = step[1]
            label = step[2].split()[0]
        start_frame = np.floor(start_time * fps)
        end_frame = np.floor(end_time * fps)
        for count in range(num_frames):
            if count >= start_frame and count <= end_frame:
                try:
                    labels[count] = int(label)
                except ValueError:
                    """
                    EGTEA annotations contains key-steps numbers as 1.
                    instead of 1
                    """
                    assert label[-1] == '.'
                    label = label.replace('.', '')
                    labels[count] = int(label)
    return labels


def eval_egoprocel(
    keystep_pred,
    keystep_gt,
    n_keystep,
    M=None,
    per_keystep=False,
    *,
    skip_background=False
):
    Z_pred = torch.eye(M)[keystep_pred.astype(np.int32), :].float().cpu().numpy()
    Z_gt = torch.eye(n_keystep)[keystep_gt, :].float().cpu().numpy()
    
    if skip_background:
        Z_gt = Z_gt[:, 1:]

    assert Z_pred.shape[0] == Z_gt.shape[0]
    T = Z_gt.shape[0]*1.0

    Dis = 1.0 - np.matmul(np.transpose(Z_gt), Z_pred)/T

    perm_gt, perm_pred = linear_sum_assignment(Dis)

    Z_pred_perm = Z_pred[:, perm_pred]
    Z_gt_perm = Z_gt[:, perm_gt]
    
    coverage = (Z_gt_perm.sum() / Z_gt_perm.shape[0])

    if per_keystep:
        list_MoF = []
        list_IoU = []
        list_precision = []
        step_wise_metrics = dict()
        for count, idx_k in enumerate(range(Z_gt_perm.shape[1])):
            pred_k = Z_pred_perm[:, idx_k]
            gt_k = Z_gt_perm[:, idx_k]

            intersect = np.multiply(pred_k, gt_k)
            union = np.clip((pred_k + gt_k).astype(float), 0, 1)

            n_intersect = np.sum(intersect)
            n_union = np.sum(union)
            n_predict = np.sum(pred_k)

            n_gt = np.sum(gt_k == 1)

            if n_gt != 0:
                MoF_k = n_intersect/n_gt
                IoU_k = n_intersect/n_union
                if n_predict == 0:
                    Prec_k = 0
                else:
                    Prec_k = n_intersect/n_predict
            else:
                MoF_k, IoU_k, Prec_k = [-1, -1, -1]
            list_MoF.append(MoF_k)
            list_IoU.append(IoU_k)
            list_precision.append(Prec_k)
            step_wise_metrics[count] = {
                "MoF": MoF_k,
                "IoU": IoU_k,
                "prec": Prec_k
            }

        arr_MoF = np.array(list_MoF)
        arr_IoU = np.array(list_IoU)
        arr_prec = np.array(list_precision)

        mask = arr_MoF != -1
        MoF = np.mean(arr_MoF[mask])
        IoU = np.mean(arr_IoU[mask])
        Precision = np.mean(arr_prec[mask])
        
        return MoF, IoU, Precision, coverage, step_wise_metrics, Z_gt_perm.argmax().astype, Z_pred_perm.argmax().astype
    
    else:
        intersect = np.multiply(Z_pred_perm, Z_gt_perm)
        union = np.clip((Z_pred_perm + Z_gt_perm).astype(float), 0, 1)

        n_intersect = np.sum(intersect)
        n_union = np.sum(union)
        n_predict = np.sum(Z_pred_perm)

        n_gt = np.sum(Z_gt_perm)

        MoF = n_intersect/n_gt
        IoU = n_intersect/n_union
        Precision = n_intersect/n_predict
    
        return MoF, IoU, Precision, coverage, None, Z_gt_perm.argmax().astype, Z_pred_perm.argmax().astype
