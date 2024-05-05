
from tensorflow.keras.models import (
    Model,
    load_model
    )
from tensorflow.keras.utils import custom_object_scope

import beliefppg.model.belief_ppg
from beliefppg.model.belief_ppg import (
    AveragePooling1D,
    FlexibleAttention,
    )
from beliefppg.model.binned_regression_loss import BinnedRegressionLoss
from beliefppg.model.positional_encoding import PositionalEncoding
from beliefppg.model.prior_layer import PriorLayer


def load_inference_model(model_path: str) -> Model:
    """
    Load a model from a given path
    :param model_path: path to the model
    :return: keras.models.Model
    """
    custom_objects = {'AveragePooling1D': AveragePooling1D,
                      'BinnedRegressionLoss': BinnedRegressionLoss,
                      'PositionalEncoding': PositionalEncoding,
                      'FlexibleAttention': FlexibleAttention,
                      'PriorLayer': PriorLayer}

    with custom_object_scope(custom_objects):
        return load_model(model_path)
