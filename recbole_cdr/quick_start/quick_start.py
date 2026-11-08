
"""
recbole_cdr.quick_start
########################
"""
import logging
from logging import getLogger
import torch

from recbole.utils import init_logger, init_seed, set_color

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.evaluator import register_hr_metric
from recbole_cdr.evaluator.test_metrics import append_metrics_csv, calculate_test_extra_metrics
from recbole_cdr.utils import get_model, get_trainer


def _torch_load_checkpoint(path, map_location=None):
    """Load a trusted local checkpoint across old and new PyTorch versions."""
    try:
        return torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_pretrained_parameters(config, model, logger=None):
    """Warm-start a training stage without restoring stale evaluation caches."""
    path = (
        config['pretrained_model_file']
        if 'pretrained_model_file' in config
        else None
    )
    checkpoint_validator = getattr(model, 'validate_pretrained_checkpoint', None)
    if not path:
        if checkpoint_validator is not None:
            checkpoint_validator(None)
        return None

    backbone_loader = getattr(
        model, 'load_pretrained_backbone_checkpoint', None
    )
    checkpoint = _torch_load_checkpoint(
        path,
        map_location='cpu' if backbone_loader is not None else config['device'],
    )
    if checkpoint_validator is not None:
        checkpoint_validator(checkpoint)
    if backbone_loader is not None:
        load_result = backbone_loader(checkpoint, path=path)
        if logger is not None:
            logger.info(
                set_color('Phase 1 DGCDRBASE checkpoint', 'yellow')
                + f': {path}'
            )
            logger.info(
                set_color('backbone_is_frozen', 'yellow')
                + f': {model.backbone_is_frozen()}'
            )
            logger.info(
                set_color('frozen representation cache', 'yellow')
                + ': ready'
            )
        return load_result

    state_dict = checkpoint.get('state_dict', checkpoint)
    strict = bool(
        config['pretrained_model_strict']
        if 'pretrained_model_strict' in config
        else True
    )
    if strict:
        load_result = model.load_state_dict(state_dict, strict=True)
        loaded_count = len(state_dict)
    else:
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state
            and tuple(value.shape) == tuple(current_state[key].shape)
        }
        load_result = model.load_state_dict(compatible_state, strict=False)
        loaded_count = len(compatible_state)
    if hasattr(model, 'init_restore_e'):
        model.init_restore_e()
    if logger is not None:
        logger.info(
            set_color('pretrained parameters', 'yellow')
            + f': {path} ({loaded_count} tensors, strict={strict})'
        )
        if not strict:
            logger.info(
                set_color('pretrained load result', 'yellow')
                + f': {load_result}'
            )
    return load_result


def _metrics_for_csv(config, best_valid_result, test_result):
    """Optionally retain the split name so grid search uses validation only."""
    include_valid = bool(
        config['metrics_csv_include_valid']
        if 'metrics_csv_include_valid' in config
        else False
    )
    if not include_valid:
        return test_result

    metrics = {}
    for key, value in (best_valid_result or {}).items():
        metrics[f'valid_{key}'] = value
    for key, value in test_result.items():
        metrics[f'test_{key}'] = value
    return metrics


def run_recbole_cdr(model=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = CDRConfig(model=model, config_file_list=config_file_list, config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    _load_pretrained_parameters(config, model, logger)
    experiment_variant = getattr(model, 'experiment_variant', None)
    if experiment_variant is not None:
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logger.info('=' * 32)
        logger.info(f'Experiment Variant: {experiment_variant}')
        logger.info(f'Trainable parameters: {trainable_parameters}')
        logger.info('=' * 32)
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # Intermediate stages only produce a checkpoint for the next stage. Running
    # full-sort test evaluation there is both expensive and statistically
    # unnecessary; the final stage evaluates the resulting model once.
    skip_test_evaluation = bool(
        config['skip_test_evaluation']
        if 'skip_test_evaluation' in config
        else False
    )
    # Parameter analysis can record popularity/diversity measures for the best
    # validation checkpoint whether or not test evaluation is also requested.
    evaluate_valid_extra = bool(
        config['evaluate_extra_metrics_on_valid']
        if 'evaluate_extra_metrics_on_valid' in config
        else False
    )
    if evaluate_valid_extra and valid_data is not None:
        analysis_valid_data = (
            valid_data[-1]
            if isinstance(valid_data, (tuple, list))
            else valid_data
        )
        evaluated_valid_result = trainer.evaluate(
            analysis_valid_data,
            load_best_model=saved,
            show_progress=config['show_progress'],
        )
        extra_valid_result = calculate_test_extra_metrics(
            config,
            model,
            train_data.dataset,
            analysis_valid_data,
        )
        if extra_valid_result:
            evaluated_valid_result.update(extra_valid_result)
        best_valid_result = evaluated_valid_result
        logger.info(
            set_color('validation analysis result', 'yellow')
            + f': {best_valid_result}'
        )

    if skip_test_evaluation:
        test_result = {}
        logger.info(set_color('test evaluation', 'yellow') + ': skipped')
        # Hyperparameter searches select trials from validation only. Persist
        # the best validation row when requested, without touching test data.
        include_valid = bool(
            config['metrics_csv_include_valid']
            if 'metrics_csv_include_valid' in config
            else False
        )
        if include_valid and best_valid_result:
            csv_path = append_metrics_csv(
                config,
                _metrics_for_csv(config, best_valid_result, test_result),
            )
            if csv_path:
                logger.info(set_color('metrics csv', 'yellow') + f': {csv_path}')
    else:
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config['show_progress']
        )
        extra_test_result = calculate_test_extra_metrics(
            config, model, train_data.dataset, test_data
        )
        if extra_test_result:
            test_result.update(extra_test_result)
        # Always persist the standard test metrics. Extra popularity/diversity
        # metrics are included when enabled, but their absence must not suppress
        # the CSV row entirely.
        csv_path = append_metrics_csv(
            config,
            _metrics_for_csv(config, best_valid_result, test_result),
        )
        if csv_path:
            logger.info(set_color('metrics csv', 'yellow') + f': {csv_path}')

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = CDRConfig(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    _load_pretrained_parameters(config, model)
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)
    extra_test_result = calculate_test_extra_metrics(config, model, train_data.dataset, test_data)
    if extra_test_result:
        test_result.update(extra_test_result)
    append_metrics_csv(
        config,
        _metrics_for_csv(config, best_valid_result, test_result),
    )

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    # 此处重新设置了CUDA! 需要与原始训练模型做映射 old-CUDA:current-CUDA
    checkpoint = _torch_load_checkpoint(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def evaluate_checkpoint_on_test(
    model_file,
    config_overrides=None,
    prefix_test_metrics=True,
):
    """Evaluate a saved training checkpoint on the test split only.

    The original seed and split configuration stored in the checkpoint are
    reused.  No optimizer step or validation evaluation is performed.
    ``config_overrides`` is intended for runtime-only settings such as the
    device and a separate metrics CSV path.
    """
    checkpoint = _torch_load_checkpoint(model_file, map_location="cpu")
    # Checkpoint-only evaluation bypasses CDRConfig.__init__, which normally
    # installs custom evaluator aliases such as explicit HR.
    register_hr_metric()
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(
            "Test-only evaluation requires a complete training checkpoint "
            "containing config and state_dict."
        )

    config = checkpoint["config"]
    compatibility_settings = getattr(config, "compatibility_settings", None)
    if compatibility_settings is not None:
        compatibility_settings()
    for key, value in (config_overrides or {}).items():
        config[key] = value

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    logger.info(
        set_color("test-only checkpoint", "yellow") + f": {model_file}"
    )

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, _, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(
        config, train_data.dataset
    ).to(config["device"])

    # PROPOSED needs its Phase 1 backbone loaded before its trainer verifies
    # the frozen cache.  The complete Phase 2 checkpoint is restored after
    # trainer construction so trained projections are not mistaken for their
    # required identity initialization.
    _load_pretrained_parameters(config, model, logger)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(
        config, model
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    # fit() normally registers training-set metadata in the evaluator.  Do
    # only that bookkeeping here; no training or validation pass is run.
    trainer.eval_collector.data_collect(train_data)
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )
    extra_test_result = calculate_test_extra_metrics(
        config, model, train_data.dataset, test_data
    )
    if extra_test_result:
        test_result.update(extra_test_result)

    csv_metrics = (
        {f"test_{key}": value for key, value in test_result.items()}
        if prefix_test_metrics
        else test_result
    )
    csv_path = append_metrics_csv(config, csv_metrics)
    if csv_path:
        logger.info(set_color("test-only metrics csv", "yellow") + f": {csv_path}")
    logger.info(set_color("test-only result", "yellow") + f": {test_result}")
    return {
        "checkpoint": model_file,
        "test_result": test_result,
        "metrics_csv": csv_path,
    }
