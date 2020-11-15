from segmentation_models import Unet, FPN
import tensorflow as tf


def get_model(cfg, training=True):
    tf.keras.backend.set_learning_phase(training)

    model = None
    n_classes = len(cfg.CLASSES.keys())
    if cfg.model_type == "UNET":
        model = Unet(
            backbone_name=cfg.backbone_name, 
            input_shape=cfg.input_shape,
            classes=n_classes, 
            activation='sigmoid' if n_classes == 1 else 'softmax',
            weights=None,
            encoder_weights = cfg.encoder_weights,
            encoder_freeze = cfg.encoder_freeze,
            encoder_features = cfg.encoder_features,
            decoder_block_type = cfg.decoder_block_type,
            decoder_filters = cfg.decoder_filters, 
            decoder_use_batchnorm=True
        )
    elif cfg.model_type == "FPN":
        model = FPN(
            backbone_name=cfg.backbone_name, 
            input_shape=cfg.input_shape,
            classes=n_classes, 
            activation='sigmoid' if n_classes == 1 else 'softmax',
            weights=None,
            encoder_weights = cfg.encoder_weights,
            encoder_freeze = cfg.encoder_freeze,
            encoder_features = cfg.encoder_features
        )
    else:
        print("Unsupported model type!")
        exit(1)

    if cfg.pretrained_model is not None:
        model.load_weights(cfg.pretrained_model)

    return model