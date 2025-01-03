from typing import List, Dict, Any, Optional
from collections import deque
import copy
import torch

from ding.utils import list_split, lists_to_dicts


def get_train_sample_spine(
        data: List[Dict[str, Any]],
        unroll_len: int,
        overlap: int = 0,
        last_fn_type: str = 'last',
        null_transition: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Overview:
        Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
        If ``unroll_len`` equals to 1, which means no process is needed, can directly return ``data``.
        Otherwise, ``data`` will be splitted according to ``unroll_len``, process residual part according to
        ``last_fn_type`` and call ``lists_to_dicts`` to form sampled training data.
    Arguments:
        - data (:obj:`List[Dict[str, Any]]`): Transitions list, each element is a transition dict
        - unroll_len (:obj:`int`): Learn training unroll length
        - last_fn_type (:obj:`str`): The method type name for dealing with last residual data in a traj \
            after splitting, should be in ['last', 'drop', 'null_padding']
        - null_transition (:obj:`Optional[dict]`): Dict type null transition, used in ``null_padding``
    Returns:
        - data (:obj:`List[Dict[str, Any]]`): Transitions list processed after unrolling
    """
    if unroll_len == 1:
        return data
    else:
        # cut data into pieces whose length is unroll_len
        split_data, residual = list_split(data, step=unroll_len, overlap= overlap)

        def null_padding():
            template = copy.deepcopy(residual[0])
            template['null'] = True  # TODO(pu)
            template['obs']['agent_state'] = torch.zeros_like(template['obs']['agent_state'])
            template['obs']['global_state'] = torch.zeros_like(template['obs']['global_state'])
            # template['action'] = -1 * torch.ones_like(template['action']) # TODO(pu)
            template['action'] = torch.zeros_like(template['action'])
            template['done'] = True
            template['reward'] = torch.zeros_like(template['reward'])
            if 'value_gamma' in template:
                template['value_gamma'] = 0.
            null_data = [_get_null_transition(template, null_transition) for _ in range(miss_num)]
            return null_data

        if residual is not None:
            miss_num = unroll_len - len(residual)
            if last_fn_type == 'drop':
                # drop the residual part
                pass
            elif last_fn_type == 'last':
                if len(split_data) > 0:
                    # copy last datas from split_data's last element, and insert in front of residual
                    last_data = copy.deepcopy(split_data[-1][-miss_num:])
                    split_data.append(last_data + residual)
                else:
                    # get null transitions using ``null_padding``, and insert behind residual
                    null_data = null_padding()
                    split_data.append(residual + null_data)
            elif last_fn_type == 'null_padding':
                # same to the case of 'last' type and split_data is empty
                null_data = null_padding()
                split_data.append(residual + null_data)
        # collate unroll_len dicts according to keys
        if len(split_data) > 0:
            split_data = [lists_to_dicts(d, recursive=True) for d in split_data]
        return split_data


def _get_null_transition(template: dict, null_transition: Optional[dict] = None) -> dict:
    """
    Overview:
        Get null transition for padding. If ``cls._null_transition`` is None, return input ``template`` instead.
    Arguments:
        - template (:obj:`dict`): The template for null transition.
        - null_transition (:obj:`Optional[dict]`): Dict type null transition, used in ``null_padding``
    Returns:
        - null_transition (:obj:`dict`): The deepcopied null transition.
    """
    if null_transition is not None:
        return copy.deepcopy(null_transition)
    else:
        return copy.deepcopy(template)