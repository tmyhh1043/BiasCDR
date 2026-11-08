"""Custom evaluator extensions for recbole_cdr."""

from recbole_cdr.evaluator.arp import register_arp_metric, register_gini_metric
from recbole_cdr.evaluator.hr import register_hr_metric

__all__ = ["register_arp_metric", "register_gini_metric", "register_hr_metric"]
