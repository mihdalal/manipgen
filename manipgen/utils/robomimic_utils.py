from copy import deepcopy
import json

from manipgen_robomimic.algo import algo_factory
from manipgen_robomimic.config import config_factory
import manipgen_robomimic.utils.file_utils as FileUtils
import manipgen_robomimic.utils.obs_utils as ObsUtils
import manipgen_robomimic.utils.train_utils as TrainUtils


def load_config(algo_cfg_path, override_cfg):
    base_cfg = json.load(open(f"config/dagger/robomimic/base_{override_cfg.dagger.visual_obs_type}.json", 'r'))
    algo_cfg = json.load(open(algo_cfg_path, 'r'))
    config = config_factory(algo_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        # load json cfg values
        config.update(base_cfg)
        config.update(algo_cfg)
        # override other params from dagger cfg
        config.experiment.name = override_cfg.exp_name
        config.train.seed = override_cfg.seed
        config.train.cuda = "cuda" in override_cfg.rl_device
        if override_cfg.dagger.batch_size is not None:
            config.train.batch_size = override_cfg.dagger.batch_size
        if override_cfg.dagger.lr is not None:
            config.algo.optim_params.policy.learning_rate.initial = override_cfg.dagger.lr

    return config


def initialize(config):
    ObsUtils.initialize_obs_utils_with_config(config)


def get_obs_shape_meta(config, dataset_spec, in_memory=True):
    dataset_path, in_memory_dataset = (dataset_spec, None) if not in_memory else (None, dataset_spec)
    return FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        in_memory_dataset=in_memory_dataset,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )


def get_dataset_loader(config, shape_meta, in_memory=True):
    if in_memory:
        return lambda in_mem_data: TrainUtils.dataset_factory(
            config=config,
            obs_keys=shape_meta["all_obs_keys"],
            filter_by_attribute=None,
            in_memory_data=in_mem_data,
        )
    else:
        return lambda path: TrainUtils.dataset_factory(
            config=config,
            obs_keys=shape_meta["all_obs_keys"],
            filter_by_attribute=None,
            dataset_path=path,
        )


def build_model(config, shape_meta, device):
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model


def run_epoch(model, data_loader, epoch, mode, n_iter=1):
    assert mode in ('train', 'eval')
    train = mode == 'train'

    info = TrainUtils.run_epoch(
        model=model,
        data_loader=data_loader,
        epoch=epoch,
        validate=not train,
        num_steps=len(data_loader) * n_iter,  # loop through dataset n_iter times
        obs_normalization_stats=None,
    )
    if train:
        model.on_epoch_end(epoch)

    return info


_hidden_state_attributes = [
    "_rnn_hidden_state"
]

def get_hidden_state(model):
    return deepcopy(tuple(getattr(model, name, None) for name in _hidden_state_attributes))


def set_hidden_state(model, state):
    for name, value in zip(_hidden_state_attributes, state):
        if hasattr(model, name):
            setattr(model, name, value)


def get_rollout_action(model, state_obs, visual_obs, state_frame0_obs, visual_frame0_obs):
    return model.get_action(
        obs_dict={
            "state": state_obs[:, :7],                  # only keep first 7 elements (proprio obs)
            "state_frame0": state_frame0_obs[:, :7],    # only keep first 7 elements (proprio obs)
            "visual": visual_obs,
            "visual_frame0": visual_frame0_obs,
        },
    )
