import os
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import onnx
import onnxruntime as ort

from datasets.datasets import Datasets
from utils.logger import setup_logger
from config.simclr_config import SimCLRConfig
from models.simclr import SimCLR


def setup_parser():
    parser = argparse.ArgumentParser(description="Convert SimCLR encoder to ONNX for embedding extraction")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("model", help="Path to model directory (checkpoint folder)")
    parser.add_argument("epoch_num", help="Epoch number to load the checkpoint from")
    return parser


def remove_prefix(state_dict, prefix="encoder."):
    """Retire le préfixe 'encoder.' de toutes les clés du state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def save_onnx_model(torch_model, config, current_epoch):
    logger = logging.getLogger(config.base.logger_name)
    # On sauvegarde avec l'extension .onnx pour plus de clarté
    onnx_model_file_path = os.path.join(config.base.log_dir_path, "checkpoint_{}.onnx".format(current_epoch))

    # Création d'un input aléatoire adapté aux dimensions d'entrée
    x = torch.randn(config.fine_tuning.batch_size, 3, config.fine_tuning.img_size,
                    config.fine_tuning.img_size, requires_grad=True)
    logger.info('Created random input for ONNX export.')

    # Teste une passe avant
    torch_out = torch_model(x)
    logger.info('Tested model output on random input.')

    torch.onnx.export(
        torch_model,               
        x,                          
        onnx_model_file_path,       
        export_params=True,        
        opset_version=10,          
        do_constant_folding=True,  
        input_names=['input'],     
        output_names=['output'],   
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )

    logger.info('Saved ONNX model: {}'.format(onnx_model_file_path))
    return onnx_model_file_path


def load_onnx_model(config, onnx_model_file_path):
    onnx_model = onnx.load(onnx_model_file_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    return onnx_model


def load_torch_model_encoder(config):
    """
    Charge le modèle SimCLR pré-entraîné et supprime la tête de projection pour ne conserver que l'encodeur.
    """
    logger = logging.getLogger(config.base.logger_name)
    
    torch_model = SimCLR.get_resnet_model(config.simclr.model.resnet)
    logger.info("Loaded auto-supervised model (with projection head).")
    
    model_file_path = os.path.join(config.onnx.model_path, "checkpoint_{}.pth".format(config.onnx.epoch_num))
    if not os.path.exists(model_file_path):
        raise FileNotFoundError('Invalid model_file_path: {}'.format(model_file_path))
    
    checkpoint = torch.load(model_file_path, map_location=config.base.device.type)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Retire le préfixe "encoder." si présent
    state_dict = remove_prefix(state_dict, prefix="encoder.")
    torch_model.load_state_dict(state_dict, strict=False)
    logger.info("State dict loaded into auto-supervised model.")
    
    # Pour extraire les embeddings, on retire la tête de projection.
    # Par exemple, si la tête de projection est contenue dans la couche fc, on la remplace par Identity.
    if hasattr(torch_model, "fc"):
        torch_model.fc = torch.nn.Identity()
        logger.info("Projection head removed: model now outputs embeddings.")
    
    return torch_model


def main(args):
    config_yaml = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if not os.path.exists(args.config):
        raise FileNotFoundError('Provided config file does not exist: %s' % args.config)

    config_yaml['logger_name'] = 'onnx'
    config = SimCLRConfig(config_yaml)

    if not os.path.exists(config.base.output_dir_path):
        os.mkdir(config.base.output_dir_path)
    if not os.path.exists(config.base.log_dir_path):
        os.makedirs(config.base.log_dir_path)

    logger = setup_logger(config.base.logger_name, config.base.log_file_path)
    logger.info('Using config: %s' % config)

    if not os.path.exists(args.model):
        raise FileNotFoundError('Provided model directory does not exist: %s' % args.model)
    else:
        logger.info('Using model directory: %s' % args.model)

    config.onnx.model_path = args.model
    logger.info('Using model_path: {}'.format(config.onnx.model_path))

    config.onnx.epoch_num = args.epoch_num
    logger.info('Using epoch_num: {}'.format(config.onnx.epoch_num))

    model_file_path = Path(config.onnx.model_path).joinpath('checkpoint_' + config.onnx.epoch_num + '.pth')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError('Model file does not exist: %s' % model_file_path)
    else:
        logger.info('Using model file: %s' % model_file_path)

    train_dataset, val_dataset, test_dataset, classes = Datasets.get_datasets(config)

    torch_model = load_torch_model_encoder(config)
    
    torch_model = torch_model.to(torch.device('cpu'))
    
    onnx_model_file_path = save_onnx_model(torch_model, config=config, current_epoch=config.onnx.epoch_num)
    
    onnx_model = load_onnx_model(config, onnx_model_file_path)
    if onnx_model:
        logger.info('Loaded ONNX model: {}'.format(onnx_model_file_path))
    

if __name__ == '__main__':
    np.random.seed(0)
    main(setup_parser().parse_args())
