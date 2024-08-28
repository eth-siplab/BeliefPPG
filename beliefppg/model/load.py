
import numpy as np
from tensorflow.keras.models import (
    Model,
    load_model,
    model_from_json
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
        with open(model_path + '_architecture.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        model_weights = np.load(model_path + '_weights.npz')
        weights_list = [model_weights[key] for key in model_weights]
        model.set_weights(weights_list)

        return model
