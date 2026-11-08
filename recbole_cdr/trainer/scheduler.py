"""Validation-only plateau scheduling shared by cross-domain trainers."""

import math

import torch


def config_value(config, name, default):
    return config[name] if name in config and config[name] is not None else default


class ValidationSchedulerMixin:
    def _init_scheduler(self):
        self.lr_scheduler = None
        self._resume_training_phase = None
        kind = config_value(self.config, "lr_scheduler", "none")
        if kind not in {"none", "plateau"}:
            raise ValueError("lr_scheduler must be none or plateau.")
        if kind == "none":
            return
        metric = config_value(self.config, "scheduler_metric", "NDCG@10").lower()
        mode = config_value(self.config, "scheduler_mode", "max")
        factor = config_value(self.config, "scheduler_factor", 0.5)
        patience = config_value(self.config, "scheduler_patience", 5)
        min_lr = config_value(self.config, "scheduler_min_lr", 1e-6)
        if mode not in {"min", "max"}:
            raise ValueError("scheduler_mode must be min or max.")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 < factor < 1:
            raise ValueError("scheduler_factor must be in (0, 1).")
        if isinstance(patience, bool) or not isinstance(patience, int) or not 0 <= patience < self.stopping_step:
            raise ValueError("scheduler_patience must be non-negative and shorter than stopping_step.")
        if isinstance(min_lr, bool) or not isinstance(min_lr, (float, int)) or not math.isfinite(min_lr) or min_lr < 0:
            raise ValueError("scheduler_min_lr must be finite and non-negative.")
        if min_lr >= self.learning_rate:
            raise ValueError("scheduler_min_lr must be lower than learning_rate.")
        try:
            name, cutoff = metric.split("@")
            if name not in {str(value).lower() for value in self.config["metrics"]}:
                raise ValueError
            if int(cutoff) not in self.config["topk"]:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"Scheduler metric {metric} must appear in metrics/topk.") from None
        self.scheduler_metric = metric
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=mode, factor=float(factor), patience=patience, min_lr=float(min_lr))

    def _valid_epoch(self, *args, **kwargs):
        score, result = super()._valid_epoch(*args, **kwargs)
        if self.lr_scheduler is not None:
            if self.scheduler_metric not in result:
                raise ValueError(f"Validation did not return {self.scheduler_metric}.")
            value = float(result[self.scheduler_metric])
            if not math.isfinite(value):
                raise ValueError("Non-finite validation metric passed to scheduler.")
            self.lr_scheduler.step(value)
        self.logger.info("Validation learning rate: %s", [g["lr"] for g in self.optimizer.param_groups])
        return score, result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        path = kwargs.pop("saved_model_file", self.saved_model_file)
        state = dict(config=self.config, epoch=epoch, cur_step=self.cur_step,
                     best_valid_score=self.best_valid_score, state_dict=self.model.state_dict(),
                     other_parameter=self.model.other_parameter(), optimizer=self.optimizer.state_dict(),
                     best_valid_result=self.best_valid_result,
                     training_phase=getattr(self, "_training_phase", 0),
                     scheduler=None if self.lr_scheduler is None else self.lr_scheduler.state_dict())
        torch.save(state, path)
        if verbose:
            self.logger.info("Saving current: %s", path)

    def resume_checkpoint(self, resume_file):
        super().resume_checkpoint(resume_file)
        state = torch.load(resume_file, map_location=self.device)
        saved_scheduler = state.get("scheduler")
        if self.lr_scheduler is not None:
            if saved_scheduler is None:
                raise ValueError("Checkpoint has no scheduler state; disable scheduler or start a new run.")
            self.lr_scheduler.load_state_dict(saved_scheduler)
        elif saved_scheduler is not None:
            raise ValueError("Checkpoint used plateau scheduling; resume with lr_scheduler: plateau.")
        self.best_valid_result = state.get("best_valid_result")
        self._resume_training_phase = state.get("training_phase", 0)
