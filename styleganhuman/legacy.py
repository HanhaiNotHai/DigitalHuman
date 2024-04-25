import copy
import pickle

import torch

import dnnlib
from torch_utils import misc


# ----------------------------------------------------------------------------
## loading torch pkl
def load_network_pkl(f, force_fp16=False, G_only=False):
    data = _LegacyUnpickler(f).load()

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
    if 'augment_pipe' not in data:
        data['augment_pipe'] = None

    # Validate contents.
    assert isinstance(data['G_ema'], torch.nn.Module)
    if not G_only:
        assert isinstance(data['D'], torch.nn.Module)
        assert isinstance(data['G'], torch.nn.Module)
        assert isinstance(data['training_set_kwargs'], (dict, type(None)))
        assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

    # Force FP16.
    if force_fp16:
        if G_only:
            convert_list = ['G_ema']  #'G'
        else:
            convert_list = ['G', 'D', 'G_ema']
        for key in convert_list:
            old = data[key]
            kwargs = copy.deepcopy(old.init_kwargs)
            if key.startswith('G'):
                kwargs.synthesis_kwargs = dnnlib.EasyDict(
                    kwargs.get('synthesis_kwargs', {})
                )
                kwargs.synthesis_kwargs.num_fp16_res = 4
                kwargs.synthesis_kwargs.conv_clamp = 256
            if key.startswith('D'):
                kwargs.num_fp16_res = 4
                kwargs.conv_clamp = 256
            if kwargs != old.init_kwargs:
                new = type(old)(**kwargs).eval().requires_grad_(False)
                misc.copy_params_and_buffers(old, new, require_all=True)
                data[key] = new
    return data


class _TFNetworkStub(dnnlib.EasyDict):
    pass


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        return super().find_class(module, name)
