"""Step Grounding eval functions taken from ego4d-goalstep/step_grounding/NaQ/VSLNet/utils/evaluate_ego4d_nlq.py"""

import numpy as np
import terminaltables


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def evaluate_nlq_performance(predictions, ground_truth, thresholds, topK, per_instance=False):
    """Evalutes the performances."""
    gt_dict = {}
    num_gt_queries = 0

    for video_datum in ground_truth["videos"]:
        for clip_datum in video_datum["clips"]:
            clip_uid = clip_datum["clip_uid"]
            for ann_datum in clip_datum["annotations"]:
                key = (clip_uid, ann_datum["annotation_uid"])
                gt_dict[key] = ann_datum
                num_gt_queries += len(ann_datum["language_queries"])

    results = [[[] for _ in topK] for _ in thresholds]
    average_IoU = []
    num_instances = 0
    for pred_datum in predictions:
        key = (pred_datum["clip_uid"], pred_datum["annotation_uid"])
        assert key in gt_dict, "Instance not present!"
        query_id = pred_datum["query_idx"]
        gt_datum = gt_dict[key]
        gt_query_datum = gt_datum["language_queries"][query_id]

        # Compute overlap and recalls.
        overlap = compute_IoU(
            pred_datum["predicted_times"],
            [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
        )
        average_IoU.append(np.mean(np.sort(overlap[0])[-3:]))
        for tt, threshold in enumerate(thresholds):
            for rr, KK in enumerate(topK):
                results[tt][rr].append((overlap > threshold)[:KK].any())
        num_instances += 1

    mean_results = np.array(results).mean(axis=-1)
    mIoU = np.mean(average_IoU)
    print(f"Evaluated: {num_instances} / {num_gt_queries} instances")

    # mean_results has shape (3, 3): axis 0 rank, axis 1 iou thresholds
    return mean_results, mIoU


def display_results(results, mIoU, thresholds, topK, title=None):
    display_data = [[f"Rank@{ii}\nmIoU@{jj}" for ii in topK for jj in thresholds] + ["mIoU"]]
    results_scaled = results * 100
    mIoU_scaled = mIoU * 100
    display_data.append([f"{results_scaled[jj][ii]:.02f}" for ii in range(len(topK)) for jj in range(len(thresholds))] + [f"{mIoU_scaled:.02f}"])
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(thresholds) * len(topK)):
        table.justify_columns[ii] = "center"
    return table.table
