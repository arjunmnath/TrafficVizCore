"""
reid/eval_metrics.py

Evaluation metrics calculator for ReID (Rank-1, Rank-5, mAP, mINP)
and Multi-Object Tracking (IDF1, HOTA, DetA, AssA, MOTA, IDSW).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional
from lap import lapjv


def compute_reid_retrieval_metrics(
    query_embeddings: np.ndarray,
    query_pids: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_pids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute ReID retrieval metrics: Rank-1, Rank-5, mAP, mINP.

    Args:
        query_embeddings: (Nq, D) array of query feature vectors.
        query_pids: (Nq,) array of query identity labels.
        gallery_embeddings: (Ng, D) array of gallery feature vectors.
        gallery_pids: (Ng,) array of gallery identity labels.
        query_cams: Optional (Nq,) array of query camera IDs.
        gallery_cams: Optional (Ng,) array of gallery camera IDs.

    Returns:
        Dict containing rank1, rank5, mAP, mINP (percentages 0-100).
    """
    Nq = query_embeddings.shape[0]
    Ng = gallery_embeddings.shape[0]

    if Nq == 0 or Ng == 0:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "mINP": 0.0}

    # L2-normalize embeddings
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix (Nq, Ng)
    sim_matrix = q_norm @ g_norm.T

    rank1_count = 0
    rank5_count = 0
    aps = []
    inps = []

    for i in range(Nq):
        q_pid = query_pids[i]
        q_cam = query_cams[i] if query_cams is not None else None

        sims = sim_matrix[i].copy()

        # Mask out self-matches (same item in query and gallery)
        if query_cams is not None and gallery_cams is not None:
            invalid = (gallery_pids == q_pid) & (gallery_cams == q_cam)
            if np.all(invalid):
                continue

        # Sort gallery items by similarity descending
        indices = np.argsort(-sims)

        matches = (gallery_pids[indices] == q_pid).astype(np.int32)
        total_gt = np.sum(matches)
        if total_gt == 0:
            continue

        # Rank-1 & Rank-5
        first_match_rank = np.where(matches == 1)[0][0] + 1
        if first_match_rank == 1:
            rank1_count += 1
        if first_match_rank <= 5:
            rank5_count += 1

        # Average Precision (AP)
        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cum_matches * matches / ranks
        ap = np.sum(precisions) / total_gt
        aps.append(ap)

        # Inverse Negative Penalty (INP)
        last_match_rank = np.where(matches == 1)[0][-1] + 1
        inp = total_gt / last_match_rank
        inps.append(inp)

    num_valid_queries = max(1, len(aps))
    rank1 = (rank1_count / num_valid_queries) * 100.0
    rank5 = (rank5_count / num_valid_queries) * 100.0
    mAP = float(np.mean(aps)) * 100.0 if aps else 0.0
    mINP = float(np.mean(inps)) * 100.0 if inps else 0.0

    return {
        "rank1": rank1,
        "rank5": rank5,
        "mAP": mAP,
        "mINP": mINP,
    }


def compute_mot_tracking_metrics(
    gt_frames: List[Dict[str, Any]],
    pred_frames: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Multi-Object Tracking metrics: IDF1, HOTA, DetA, AssA, MOTA, IDSW.

    Args:
        gt_frames: List of dicts per frame with 'boxes' (N, 4) and 'ids' (N,).
        pred_frames: List of dicts per frame with 'boxes' (M, 4) and 'ids' (M,).
        iou_threshold: IoU threshold for spatial bounding box match.

    Returns:
        Dict of metrics (IDF1, HOTA, DetA, AssA, MOTA, IDSW).
    """

    def compute_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / (union_area + 1e-8)

    num_frames = max(len(gt_frames), len(pred_frames))
    total_gt = 0
    total_pred = 0

    frame_matches = []  # List of (frame_idx, gt_id, pred_id, iou)

    for f_idx in range(num_frames):
        gt = gt_frames[f_idx] if f_idx < len(gt_frames) else {"boxes": [], "ids": []}
        pred = pred_frames[f_idx] if f_idx < len(pred_frames) else {"boxes": [], "ids": []}

        gt_boxes = gt.get("boxes", [])
        gt_ids = gt.get("ids", [])
        pred_boxes = pred.get("boxes", [])
        pred_ids = pred.get("ids", [])

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        if not gt_boxes or not pred_boxes:
            continue

        cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
        for i, g_box in enumerate(gt_boxes):
            for j, p_box in enumerate(pred_boxes):
                iou = compute_iou(g_box, p_box)
                if iou >= iou_threshold:
                    cost_matrix[i, j] = 1.0 - iou

        # Hungarian assignment via lapjv
        opt_cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=1.0 - iou_threshold)
        for r, c in enumerate(x):
            if c >= 0 and cost_matrix[r, c] < (1.0 - iou_threshold):
                iou = 1.0 - cost_matrix[r, c]
                frame_matches.append((f_idx, gt_ids[r], pred_ids[c], iou))

    gt_id_counts: Dict[int, int] = {}
    pred_id_counts: Dict[int, int] = {}
    pair_counts: Dict[Tuple[int, int], int] = {}

    for _, g_id, p_id, _ in frame_matches:
        gt_id_counts[g_id] = gt_id_counts.get(g_id, 0) + 1
        pred_id_counts[p_id] = pred_id_counts.get(p_id, 0) + 1
        pair_counts[(g_id, p_id)] = pair_counts.get((g_id, p_id), 0) + 1

    gt_unique_ids = list(gt_id_counts.keys())
    pred_unique_ids = list(pred_id_counts.keys())

    idtp = 0
    if gt_unique_ids and pred_unique_ids:
        cost = np.zeros((len(gt_unique_ids), len(pred_unique_ids)), dtype=np.float32)
        for i, g_id in enumerate(gt_unique_ids):
            for j, p_id in enumerate(pred_unique_ids):
                overlap = pair_counts.get((g_id, p_id), 0)
                cost[i, j] = -overlap

        opt_cost, x, y = lapjv(cost, extend_cost=True)
        for r, c in enumerate(x):
            if c >= 0:
                idtp += pair_counts.get((gt_unique_ids[r], pred_unique_ids[c]), 0)

    idfp = total_pred - idtp
    idfn = total_gt - idtp

    idf1 = (2.0 * idtp / (2.0 * idtp + idfp + idfn + 1e-8)) * 100.0
    idp = (idtp / (idtp + idfp + 1e-8)) * 100.0
    idr = (idtp / (idtp + idfn + 1e-8)) * 100.0

    tp_count = len(frame_matches)
    fp_count = total_pred - tp_count
    fn_count = total_gt - tp_count

    deta = (tp_count / (tp_count + fp_count + fn_count + 1e-8)) * 100.0
    assa = (idtp / (tp_count + 1e-8)) * 100.0
    hota = np.sqrt(deta * assa)

    mota = max(0.0, (1.0 - (fp_count + fn_count) / (total_gt + 1e-8))) * 100.0

    return {
        "IDF1": float(idf1),
        "IDP": float(idp),
        "IDR": float(idr),
        "HOTA": float(hota),
        "DetA": float(deta),
        "AssA": float(assa),
        "MOTA": float(mota),
        "IDTP": int(idtp),
        "IDFP": int(idfp),
        "IDFN": int(idfn),
    }
