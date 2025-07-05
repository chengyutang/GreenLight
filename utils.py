from xml.etree import ElementTree


def make_env(env_class, config_file, **kwargs):
    def _init():
        ret_env = env_class(config_file, **kwargs)
        return ret_env
    return _init


def get_lr_schedule(init_lr: float, start_decay_at: float = 1.0, end_lr: float = 0.0):
    if start_decay_at == 0:
        def _lr_schedule(progress_remaining: float):
            return init_lr
    else:
        def _lr_schedule(progress_remaining: float):
            return end_lr + (init_lr - end_lr) * min(1.0, progress_remaining / start_decay_at)
    return _lr_schedule
