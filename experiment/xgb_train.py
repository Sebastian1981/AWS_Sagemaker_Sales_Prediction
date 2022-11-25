import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.7)
    parser.add_argument('--objective', type=str, default='reg:squarederror')
    parser.add_argument('--num_round', type=int, default=50)
    
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args = parser.parse_args()

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': args.objective,
        'num_round': args.num_round
    }

    dtrain = xgb.DMatrix(args.train)
    dval = xgb.DMatrix(args.validation)
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    callbacks = []
    prev_checkpoint, n_iterations_prev_run = add_checkpointing(callbacks)
    # If checkpoint is found then we reduce num_boost_round by previously run number of iterations

    bst = xgb.train(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=(args.num_round - n_iterations_prev_run),
        xgb_model=prev_checkpoint,
        callbacks=callbacks
    )

    # Save the model to the location specified by ``model_dir``
    model_location = args.model_dir + '/xgboost-model'
    pkl.dump(bst, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))