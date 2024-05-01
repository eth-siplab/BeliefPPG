
import keras.models
from keras.saving import custom_object_scope

import beliefppg.model.belief_ppg
from beliefppg.model.binned_regression_loss import BinnedRegressionLoss
from beliefppg.model.positional_encoding import PositionalEncoding
from beliefppg.model.prior_layer import PriorLayer


def load_inference_model(model_path: str) -> keras.models.Model:
    """
    Load a model from a given path
    :param model_path: path to the model
    :return: keras.models.Model
    """
    custom_objects = {'PositionalEncoding': PositionalEncoding,
                      'BinnedRegressionLoss': BinnedRegressionLoss,
                      'PriorLayer': PriorLayer}

    with custom_object_scope(custom_objects):
        return keras.models.load_model(model_path, safe_mode=False)
    

