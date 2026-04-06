import numpy as np
from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS

@METRICS.register_module()
class DistalOKSMetric(BaseMetric):
    def __init__(self, amputation_types, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.amputation_types = amputation_types
        self.k_vars = None

    def process(self, data_batch, data_samples):
        if self.k_vars is None:
            sigmas = self.dataset_meta.get('sigmas', None)
            if sigmas is None:
                raise ValueError("Dataset METAINFO does not contain 'sigmas'!")
            self.k_vars = (np.array(sigmas) * 2) ** 2

        for data_sample in data_samples:
            pred_coords = data_sample['pred_instances']['keypoints'][0]
            gt_coords = data_sample['gt_instances']['keypoints'][0]
            weights = data_sample['gt_instances']['keypoint_weights'][0]

            active_indices = np.where(weights > 0)[0]
            if len(active_indices) == 0:
                continue

            bbox = data_sample['gt_instances']['bboxes'][0]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area = max(area, 1.0)

            oks_sum = 0.0
            for idx in active_indices:
                d_pt_idx = self.amputation_types[idx]['d']
                k_var = self.k_vars[d_pt_idx]

                d_sq = np.sum((pred_coords[idx] - gt_coords[idx]) ** 2)

                point_oks = np.exp(-d_sq / (2 * area * k_var))
                oks_sum += point_oks

            sample_oks = oks_sum / len(active_indices)

            self.results.append({
                'oks': sample_oks,
                'active_categories': active_indices.tolist()
            })

    def compute_metrics(self, results):
        oks_scores = np.array([res['oks'] for res in results])

        if len(oks_scores) == 0:
            return {'AP': 0.0, 'AP@0.5': 0.0, 'AP@0.75': 0.0}

        thresholds = np.arange(0.50, 1.00, 0.05)
        ap = np.mean([(oks_scores >= thr).mean() for thr in thresholds])

        eval_results = {
            'AP': ap * 100,
            'AP@0.5': (oks_scores >= 0.5).mean() * 100,
            'AP@0.75': (oks_scores >= 0.75).mean() * 100,
            'mean_OKS': oks_scores.mean()
        }
        return eval_results