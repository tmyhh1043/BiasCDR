"""Hit Ratio metric alias with an explicit ``hr@K`` output name."""

from __future__ import annotations


def register_hr_metric() -> bool:
    """Register HR as RecBole's Hit metric with an ``hr`` result key."""
    try:
        import recbole.evaluator.metrics as metrics
        from recbole.evaluator import (
            metric_information,
            metric_types,
            metrics_dict,
        )
    except Exception:
        return False

    if "hr" in metrics_dict:
        return True

    class HR(metrics.Hit):
        """Hit Ratio; numerically identical to RecBole's ``Hit`` metric."""

        def calculate_metric(self, dataobject):
            pos_index, _ = self.used_info(dataobject)
            return self.topk_result("hr", self.metric_info(pos_index))

    HR.__name__ = "HR"
    HR.__qualname__ = "HR"
    metrics.HR = HR
    metrics_dict["hr"] = HR
    metric_information["hr"] = HR.metric_need
    metric_types["hr"] = HR.metric_type
    return True


__all__ = ["register_hr_metric"]
