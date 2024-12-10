import torch

from algorithm.core.utils.config import configs

from algorithm.quantize.custom_quantized_format import build_quantized_network_from_cfg
from algorithm.quantize.quantize_helper import create_scaled_head, create_quantized_head

__all__ = ['build_mcu_model']


def build_mcu_model():
    cfg_path = f"/home/gridsan/agupta2/6.5940/tiny-training/algorithm/assets/mcu_models/{configs.net_config.net_name}.pkl"
    # cfg_path = f"../../../algorithm/runs/celeba/mcunet-5fps/sparse_50kb/sgd_qas_nomom/model.pkl"
    # cfg_path = f"/home/gridsan/agupta2/6.5940/tiny-training/algorithm/runs/celeba/mcunet-5fps/sparse_50kb/sgd_qas_nomom/model.pkl"
    cfg = torch.load(cfg_path)
    # print("There are ", len(cfg['blocks']), " layers in the cfg")
    # print("All layers:", [type(layer) for layer in cfg['blocks']])
    
    model = build_quantized_network_from_cfg(cfg, n_bit=8)

    model = create_quantized_head(model)
    # model = create_scaled_head(model, norm_feat=False)

    # if configs.net_config.mcu_head_type == 'quantized':
    #     model = create_quantized_head(model)
    # elif configs.net_config.mcu_head_type == 'fp':
    #     model = create_scaled_head(model, norm_feat=False)
    # else:
    #     raise NotImplementedError

    return model
