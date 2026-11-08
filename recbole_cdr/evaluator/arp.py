"""Custom metric registration for RecBole without modifying RecBole itself."""

from __future__ import annotations


def register_arp_metric() -> bool:
    """Register ARP metric into RecBole's evaluator registry.

    Returns:
        bool: True if registration succeeded, False otherwise.
    """
    try:
        from recbole.utils import EvaluatorType
        from recbole.evaluator import metric_types, smaller_metrics
        import recbole.evaluator.metrics as metrics
    except Exception:
        return False

    if hasattr(metrics, "ARP"):
        metric_types.setdefault("arp", EvaluatorType.RANKING)
        if "arp" not in smaller_metrics:
            smaller_metrics.append("arp")
        return True

    if not hasattr(metrics, "TopkMetric"):
        return False

    import numpy as np

    class ARP(metrics.TopkMetric):
        """Average Recommendation Popularity."""

        def calculate_metric(self, dataobject):
            rec_items = dataobject.get("rec.topk")
            item_count = dataobject.get("item_count")
            if item_count is None:
                item_count = dataobject.get("item_popularity")
            if item_count is None:
                result = np.zeros(len(self.topk), dtype=np.float32)
                return self.topk_result("ARP", result)

            if hasattr(rec_items, "cpu"):
                rec_items = rec_items.cpu().numpy()
            else:
                rec_items = np.asarray(rec_items)

            if hasattr(item_count, "cpu"):
                item_count = item_count.cpu().numpy()
            else:
                item_count = np.asarray(item_count)

            pop = item_count[rec_items]
            result = pop.mean(axis=0)
            return self.topk_result("ARP", result)

    metrics.ARP = ARP

    if hasattr(metrics, "metrics_dict"):
        metrics.metrics_dict["ARP"] = ARP
        metrics.metrics_dict["arp"] = ARP

    metric_types.setdefault("arp", EvaluatorType.RANKING)
    if "arp" not in smaller_metrics:
        smaller_metrics.append("arp")

    return True


def register_gini_metric() -> bool:
    """Register Gini coefficient metric into RecBole's evaluator registry.

    Returns:
        bool: True if registration succeeded, False otherwise.
    """
    try:
        from recbole.utils import EvaluatorType
        from recbole.evaluator import metric_types, smaller_metrics
        import recbole.evaluator.metrics as metrics
    except Exception:
        return False

    if hasattr(metrics, "Gini"):
        metric_types.setdefault("gini", EvaluatorType.RANKING)
        if "gini" not in smaller_metrics:
            smaller_metrics.append("gini")
        return True

    if not hasattr(metrics, "TopkMetric"):
        return False

    import numpy as np

    def _gini(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return 0.0
        total = values.sum()
        if total <= 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = sorted_vals.size
        index = np.arange(1, n + 1, dtype=np.float64)
        return (2.0 * np.sum(index * sorted_vals) / (n * total)) - (n + 1.0) / n

    class Gini(metrics.TopkMetric):
        """Gini coefficient of item popularity in the recommendation list."""

        def calculate_metric(self, dataobject):
            rec_items = dataobject.get("rec.topk")
            item_count = dataobject.get("item_count")
            if item_count is None:
                item_count = dataobject.get("item_popularity")
            if item_count is None:
                result = np.zeros(len(self.topk), dtype=np.float32)
                return self.topk_result("Gini", result)

            if hasattr(rec_items, "cpu"):
                rec_items = rec_items.cpu().numpy()
            else:
                rec_items = np.asarray(rec_items)

            if hasattr(item_count, "cpu"):
                item_count = item_count.cpu().numpy()
            else:
                item_count = np.asarray(item_count)

            pop = item_count[rec_items]
            results = []
            for k in self.topk:
                if k <= 0:
                    results.append(0.0)
                    continue
                gini_vals = [_gini(row[:k]) for row in pop]
                results.append(float(np.mean(gini_vals)))
            result = np.asarray(results, dtype=np.float32)
            return self.topk_result("Gini", result)

    metrics.Gini = Gini

    if hasattr(metrics, "metrics_dict"):
        metrics.metrics_dict["Gini"] = Gini
        metrics.metrics_dict["gini"] = Gini

    metric_types.setdefault("gini", EvaluatorType.RANKING)
    if "gini" not in smaller_metrics:
        smaller_metrics.append("gini")

    return True
