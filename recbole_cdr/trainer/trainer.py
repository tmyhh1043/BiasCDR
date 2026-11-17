
r"""
recbole_cdr.trainer.trainer
################################
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from time import time
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color
from recbole_cdr.utils import train_mode2state
from recbole_cdr.trainer.scheduler import ValidationSchedulerMixin
from recbole_cdr.evaluator.test_metrics import calculate_validation_coverage_arp


def _dataset_checkpoint_file(config):
    """Return a stable checkpoint path based on the experiment direction."""
    dataset_name = config["source_domain"]["dataset"]
    dataset_name = os.path.basename(str(dataset_name).strip())
    safe_name = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in dataset_name
    )
    if not safe_name:
        raise ValueError("The source-domain dataset name must not be empty.")
    stage_suffix = ''
    if config['model'] == 'C2DR' and 'c2dr_stage' in config:
        raw_stage = str(config['c2dr_stage']).strip()
        safe_stage = ''.join(
            character
            if character.isalnum() or character in {'-', '_', '.'}
            else '_'
            for character in raw_stage
        )
        if safe_stage:
            stage_suffix = f'-{safe_stage}'
    if config['model'] in {'DGCDR', 'P2M2CDR', 'DCCDR', 'DTCDR', 'BiTGCF'}:
        stage_suffix += f'-seed{int(config["seed"])}'
    return os.path.join(
        config["checkpoint_dir"], f"{safe_name}{stage_suffix}.pth"
    )


class CrossDomainTrainer(ValidationSchedulerMixin, Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.saved_model_file = _dataset_checkpoint_file(config)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']
        self.valid_start_epoch = int(config['valid_start_epoch'] if 'valid_start_epoch' in config else 1)
        self.record_valid_popularity_each_epoch = bool(
            config['record_valid_popularity_each_epoch']
            if 'record_valid_popularity_each_epoch' in config else False
        )
        self._init_scheduler()

    def _record_valid_epoch(self, epoch_idx, train_data, valid_data, result):
        if not self.record_valid_popularity_each_epoch:
            return
        analysis_data = valid_data[-1] if isinstance(valid_data, (tuple, list)) else valid_data
        result.update(calculate_validation_coverage_arp(
            self.config, self.model, train_data.dataset, analysis_data
        ))
        path_value = (self.config['valid_epoch_metrics_csv_path']
                      if 'valid_epoch_metrics_csv_path' in self.config else None)
        if not path_value:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            'source': self.config['source_domain']['dataset'],
            'target': self.config['target_domain']['dataset'],
            'variant': (self.config['diagnostic_variant']
                        if 'diagnostic_variant' in self.config else ''),
            'phase': self._training_phase,
            'epoch': epoch_idx,
            **result,
        }
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open('a', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _fit_with_valid_start(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        """RecBole Trainer.fit with validation skipped before valid_start_epoch.

        valid_start_epoch is 1-based: valid_start_epoch=30 means the first
        possible validation is after epoch_idx=29, i.e., epoch 30.
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
                self.logger.info('Learning rate: %s', [g['lr'] for g in self.optimizer.param_groups])
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx},
                head='train',
            )

            if self.eval_step <= 0 or not valid_data:
                if saved and self._should_save_without_validation(epoch_idx):
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            # Skip expensive validation until the requested 1-based start epoch.
            if (epoch_idx + 1) < self.valid_start_epoch:
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )
                self._record_valid_epoch(epoch_idx, train_data, valid_data, valid_result)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color('epoch %d evaluating', 'green')
                    + ' ['
                    + set_color('time', 'blue')
                    + ': %.2fs, '
                    + set_color('valid_score', 'blue')
                    + ': %f]'
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, 'valid_step': valid_step}, head='valid'
                )

                if update_flag:
                    self.best_valid_result = valid_result
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _skip_validation_for_phase(self, scheme):
        """Return whether validation should be disabled for a training phase."""
        return False

    def _should_save_without_validation(self, epoch_idx):
        """Return whether to save an epoch when no validation is performed."""
        return True

    def _valid_epoch(self, valid_data, show_progress=False):
        """Use GCN Recall only to select the GCN teacher checkpoint."""
        stage_validation_setter = getattr(
            self.model, "set_stage_validation_mode", None
        )
        use_gcn = (
            getattr(self.model, "training_stage", None) == "gcn_pretrain"
        )
        if stage_validation_setter is not None:
            stage_validation_setter(use_gcn)
        try:
            return super()._valid_epoch(
                valid_data, show_progress=show_progress
            )
        finally:
            if stage_validation_setter is not None:
                stage_validation_setter(False)

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self._training_phase = phase
        if self._resume_training_phase == phase:
            self._resume_training_phase = None
            self.epochs = int(self.train_epochs[phase])
            self.eval_step = min(self.config['eval_step'], self.epochs)
            return
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        first_phase = self._resume_training_phase or 0
        for phase in range(first_phase, len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if self._skip_validation_for_phase(scheme):
                phase_valid_data = None
            elif self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    phase_valid_data = source_valid_data
                else:
                    phase_valid_data = target_valid_data
            else:
                phase_valid_data = valid_data
            self._fit_with_valid_start(
                train_data,
                phase_valid_data,
                verbose,
                saved,
                show_progress,
                callback_fn,
            )

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result


class PROPOSEDSISTrainer(CrossDomainTrainer):
    """Train SIS heads over the frozen Phase 1 GCN representations."""

    def __init__(self, config, model):
        super().__init__(config, model)
        checkpoint_root, extension = os.path.splitext(self.saved_model_file)
        if model.mm_tail_weight:
            variant = '-M4'
        elif model.mm_kd_weight:
            variant = '-M3'
        elif model.mm_popularity_gate:
            variant = '-M2'
        elif model.mm_fusion_weight:
            variant = '-M1'
        else:
            variant = ''
        self.saved_model_file = f'{checkpoint_root}-PROPOSEDSIS{variant}{extension}'
        if not model._frozen_cache_ready:
            raise RuntimeError('Frozen GCN representation cache is not ready.')
        if model.mm_fusion_weight:
            self.logger.info(
                '%s trainable parameters: %s', variant.lstrip('-'),
                [name for name, _ in model.phase2_named_parameters()],
            )

    def _build_optimizer(self, **kwargs):
        return super()._build_optimizer(
            params=list(self.model.phase2_parameters()), **kwargs
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        output = super()._generate_train_loss_output(
            epoch_idx, s_time, e_time, losses
        )
        if not self.model.mm_fusion_weight or not isinstance(losses, tuple):
            return output
        names = (
            'tail_weighted_fusion_bpr' if self.model.mm_tail_weight else 'fusion_bpr',
            'source_observation', 'source_preference', 'cross_alignment',
            'adversarial', 'beta_regularization',
        )
        if self.model.mm_kd_weight:
            names += ('ranking_distillation',)
        decimals = int(self.config['loss_decimal_place'])
        return output + ' | ' + ', '.join(
            f'{name}: {float(value):.{decimals}f}'
            for name, value in zip(names, losses)
        )


class PROPOSEDTrainer(CrossDomainTrainer):
    """Train the end-to-end Proposed model with the standard optimizer."""

    def __init__(self, config, model):
        super().__init__(config, model)
        checkpoint_root, checkpoint_extension = os.path.splitext(
            self.saved_model_file
        )
        self.saved_model_file = (
            f'{checkpoint_root}-PROPOSED-ADDITIVE-seed{int(config["seed"])}'
            f'{checkpoint_extension}'
        )

        self.logger.info('Proposed checkpoint: %s', self.saved_model_file)


class PROPOSED_V2Trainer(CrossDomainTrainer):
    """Name and verify the independently optimized PROPOSED_V2 losses."""

    LOSS_NAMES = (
        'recommendation_loss',
        'pd_loss',
        'popularity_ranking_loss',
        'review_loss',
        'multimodal_loss',
        'alignment_loss',
        'orthogonality_loss',
        'regularization_loss',
    )

    def __init__(self, config, model):
        super().__init__(config, model)
        checkpoint_root, checkpoint_extension = os.path.splitext(
            self.saved_model_file
        )
        self.saved_model_file = (
            f'{checkpoint_root}-PROPOSED_V2{checkpoint_extension}'
        )
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group['params']
        }
        trainable_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        if optimizer_ids != trainable_ids:
            raise RuntimeError(
                'PROPOSED_V2 optimizer does not contain every trainable '
                'parameter exactly once.'
            )

    def _generate_train_loss_output(
        self, epoch_idx, s_time, e_time, losses
    ):
        output = super()._generate_train_loss_output(
            epoch_idx, s_time, e_time, losses
        )
        if not isinstance(losses, tuple) or len(losses) != len(
            self.LOSS_NAMES
        ):
            return output
        decimals = int(self.config['loss_decimal_place'])
        values = [float(loss) for loss in losses]
        named = [f'total_loss: {sum(values):.{decimals}f}']
        named.extend(
            f'{name}: {value:.{decimals}f}'
            for name, value in zip(self.LOSS_NAMES, values)
        )
        return output + ' | ' + ', '.join(named)

    def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
        epoch_setter = getattr(self.model, 'set_training_epoch', None)
        if epoch_setter is not None:
            epoch_setter(epoch_idx)
        return super()._train_epoch(
            train_data, epoch_idx, *args, **kwargs
        )


class MIVPRecTrainer(CrossDomainTrainer):
    """Cross-domain trainer for a statically selected MIVPRec stage."""

    pass


class PREFPOP_CDRTrainer(CrossDomainTrainer):
    """Use the standard cross-domain training/logging behavior."""

    pass


class C2DRTrainer(CrossDomainTrainer):
    """Use the optimizer and learning rate for one static C2DR stage."""

    def _build_optimizer(self, **kwargs):
        parameters = self.model.stage_parameters()
        return super()._build_optimizer(
            params=parameters,
            learning_rate=self.config['learning_rate'],
            **kwargs,
        )

    def _skip_validation_for_phase(self, scheme):
        return self.model.c2dr_stage == 'reweight'

    def _should_save_without_validation(self, epoch_idx):
        return self.model.c2dr_stage != 'reweight' or epoch_idx == self.epochs - 1


class DCDCSRTrainer(ValidationSchedulerMixin, Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']
        self._init_scheduler()

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self._training_phase = phase
        if self._resume_training_phase == phase:
            self._resume_training_phase = None
            self.epochs = int(self.train_epochs[phase])
            self.eval_step = min(self.config['eval_step'], self.epochs)
            return
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        first_phase = self._resume_training_phase or 0
        for phase in range(first_phase, len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result
