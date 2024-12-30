exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'retry_type': 'reset',
            'auto_reset': True,
            'step_timeout': None,
            'reset_timeout': None,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'type': 'base'
        },
        'type': 'cartpole',
        'import_names': ['dizoo.classic_control.cartpole.envs.cartpole_env'],
        'collector_env_num': 1,
        'evaluator_env_num': 1,
        'n_evaluator_episode': 1,
        'stop_value': 12,
        'act_scale': 1
    },
    'policy': {
        'model': {
            'twin_critic': False,
            'obs_shape': 14,
            'global_obs_shape': 18,
            'action_shape': 4,
            'size_list': [64, 128],
            'head_hidden_size': None,
            'head_layer_num': 2,
            'lstm_type': 'normal',
            'encoder_hidden_size_list': [128],
            'single_layer': False,
            'norm_type': 'LN'
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 2000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 100,
            'batch_size': 64,
            'learning_rate_actor': 0.0005,
            'learning_rate_critic': 0.001,
            'ignore_done': True,
            'target_theta': 0.005,
            'actor_update_freq': 10,
            'noise': True,
            'learning_rate_gamma': 0.9999,
            'max_slope': 1,
            'slope_decay': 0,
            'init_slope': 1,
            'noise_sigma': 0.05,
            'noise_range': {
                'min': -0.1,
                'max': 0.1
            }
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'get_train_sample': True,
                'cfg_type': 'AstraeaEpisodeSerialCollectorDict',
                'type': 'astraea_episode',
                'import_names': ['env.di_engine_collector'],
                'collector_num': 1,
                'stack_state': True,
                'stack_length': 1,
                'reward_length': 5,
                'use_spine': True,
                'jump_exp': 1000000,
                'env_num': 1
            },
            'n_episode': 1,
            'noise_sigma': 1.5,
            'noise_exp': 200000,
            'noise_end': 0.05,
            'random_exp': 50000,
            'env_num': 1
        },
        'eval': {
            'evaluator': {
                'eval_freq': 500,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stack_state': True,
                'stack_length': 1,
                'reward_length': 5,
                'use_spine': True,
                'type': 'astraea_episode_evaluate',
                'import_names': ['env.di_engine_evaluator'],
                'env_num': 1,
                'stop_value': 12,
                'n_episode': 1
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'naive',
                'replay_buffer_size': 10000,
                'deepcopy': False,
                'enable_track_used_data': False,
                'periodic_thruput_seconds': 60,
                'cfg_type': 'NaiveReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            },
            'eps': {
                'type': 'exp',
                'start': 0.95,
                'end': 0.05,
                'decay': 10000
            }
        },
        'type': 'spine_policy_command',
        'cuda': False,
        'on_policy': False,
        'multi_agent': False,
        'priority': False,
        'priority_IS_weight': False,
        'action_shape': 'continuous',
        'reward_batch_norm': False,
        'burnin_step': 5,
        'unroll_len': 10,
        'nstep': 1,
        'discount_factor': 0.98,
        'cfg_type': 'SpineCommandModePolicyDict',
        'import_names': ['train.policy.spine'],
        'unroll_overlap': 13
    },
    'exp_name': 'vanilla_train_seed0',
    'seed': 0
}
