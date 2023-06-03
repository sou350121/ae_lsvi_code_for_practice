import jax
import pickle
from flax import linen as nn
from flax.linen.initializers import zeros
import numpy as np
from flax.training import train_state
from jax import random, numpy as jnp, jit
from jax import tree_util as jtu
from tinygp import kernels, transforms, GaussianProcess
from tqdm import trange, tqdm
from typing import Sequence
from sklearn.model_selection import train_test_split
from hydra.utils import to_absolute_path
from copy import deepcopy
import optax

import logging

log = logging.getLogger(__name__)


class RBFGP(nn.Module):
    noise: float
    jitter: float
    is_jax: bool = True

    @nn.compact
    def __call__(self, Xtrain, Ytrain, Xtest=None, train_diag=None, test_diag=None):
        # Set up a typical RBF kernel
        log_sigma = self.param("log_sigma", zeros, ())
        log_rho = self.param("log_rho", zeros, Xtrain.shape[-1])
        kernel = jnp.exp(2 * log_sigma) * transforms.Linear(
            jnp.exp(log_rho), kernels.ExpSquared()
        )
        if train_diag is None:
            train_diag = self.noise**2 + jnp.exp(2 * jnp.log(self.jitter)) * jnp.ones(
                Ytrain.shape[0]
            )
        gp = GaussianProcess(kernel, Xtrain, diag=train_diag)
        if Xtest is not None and test_diag is None:
            test_diag = self.noise**2 + jnp.exp(2 * jnp.log(self.jitter)) * jnp.ones(
                Xtest.shape[0]
            )
        log_prob, gp_cond = gp.condition(Ytrain, X_test=Xtest, diag=test_diag)
        return log_prob, gp_cond


class Trainer:
    def __init__(
        self,
        lr,
        num_iters,
        weight_decay,
        load_params,
        seed,
        params_path=None,
        silent=False,
    ):
        self.lr = lr
        self.num_iters = num_iters
        self.weight_decay = weight_decay
        self.key = random.PRNGKey(seed)
        self.silent = silent
        if load_params:
            with open(to_absolute_path(params_path), "rb") as f:
                self.params = pickle.load(f)
        else:
            self.params = None

    def get_key(self):
        self.key, new_key = random.split(self.key)
        return new_key


class GPTrainer(Trainer):
    def __init__(
        self,
        lr,
        num_iters,
        weight_decay,
        load_params,
        seed,
        params_path=None,
        constrain_gd=False,
    ):
        super().__init__(
            lr, num_iters, weight_decay, load_params, seed, params_path=params_path
        )
        self.constrain_gd = constrain_gd
        self.log_param_min = -4
        self.log_param_max = 1

    @staticmethod
    @jit
    def _gp_train_step(state, Xtrain, Ytrain, train_diag, constraints):
        def loss_fn(params):
            logprob, _ = state.apply_fn(params, Xtrain, Ytrain, train_diag)
            return -logprob

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        if constraints is not None:
            min_constraint = lambda x: jnp.maximum(constraints[0], x)
            max_constraint = lambda x: jnp.minimum(constraints[1], x)
            new_params = jtu.tree_map(
                max_constraint, jtu.tree_map(min_constraint, state.params)
            )
            state = state.replace(params=new_params)
        return state, loss

    def train(self, gp, Xtrain, Ytrain, t, train_diag=None):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0005,
            peak_value=0.005,
            warmup_steps=50,
            decay_steps=1_000,
            end_value=0.0,
        )
        tx = optax.adamw(schedule, weight_decay=self.weight_decay)
        if self.params is not None:
            gp_params = self.params[t]
            gp_train_state = train_state.TrainState.create(
                apply_fn=gp.apply, params=gp_params, tx=tx
            )
            return gp_train_state
        gp_params = gp.init(self.get_key(), Xtrain, Ytrain)
        constraints = None
        if self.constrain_gd:
            constraints = (self.log_param_min, self.log_param_max)
        gp_train_state = train_state.TrainState.create(
            apply_fn=gp.apply,
            params=gp_params,
            tx=tx,
        )
        t = trange(self.num_iters, disable=self.silent)
        for step in t:
            gp_train_state, loss = self._gp_train_step(
                gp_train_state,
                Xtrain,
                Ytrain,
                train_diag,
                constraints,
            )
            t.set_postfix(nll=loss)
        return gp_train_state

    @staticmethod
    def _pred(X, *, train_state, Xtrain, Ytrain, train_diag):
        logprob, gp_cond = train_state.apply_fn(
            train_state.params,
            Xtrain=Xtrain,
            Ytrain=Ytrain,
            train_diag=train_diag,
            Xtest=X,
        )
        return gp_cond.loc[:, None], gp_cond.variance[:, None]


class PNN(nn.Module):
    features: Sequence[int]
    logvar_lb: float = None
    logvar_ub: float = None
    logvar_loss_coef: float = None
    noise: float = None
    jitter: float = None
    is_jax: bool = True

    @nn.compact
    def __call__(
        self, x, return_logvar=True, return_last_representation=False, train_mode=False
    ):
        features = list(self.features) + [1]
        for feat in features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        last_representation = x
        x = nn.Dense(features[-1] * 2)(x)
        mu, logvar = x[..., : features[-1]], x[..., features[-1] :]
        if self.logvar_lb is not None:
            logvar_lb = self.param(
                "logvar_lb_param", jax.nn.initializers.constant(self.logvar_lb), ()
            )
            logvar_ub = self.param(
                "logvar_ub_param", jax.nn.initializers.constant(self.logvar_ub), ()
            )
            logvar = logvar_ub - jax.nn.softplus(logvar_ub - logvar)
            logvar = logvar_lb + jax.nn.softplus(logvar - logvar_lb)
        else:
            logvar_lb = None
            logvar_ub = None
        if train_mode:
            return mu, logvar, logvar_lb, logvar_ub, self.logvar_loss_coef
        elif return_logvar and not return_last_representation:
            return mu, logvar
        elif return_logvar and return_last_representation:
            return mu, logvar, last_representation
        elif not return_logvar and return_last_representation:
            return mu, last_representation
        else:
            return mu


class PNNTrainer(Trainer):
    def __init__(
        self,
        lr,
        num_iters,
        weight_decay,
        batch_size,
        load_params,
        seed,
        params_path=None,
    ):
        super().__init__(
            lr, num_iters, weight_decay, load_params, seed, params_path=params_path
        )
        self.batch_size = batch_size

    @staticmethod
    @jax.jit
    def _train_step(state, Xbatch, Ybatch):
        """
        Train for a single step.
        """

        def loss_fn(params):
            yhat, logvar, logvar_lb, logvar_ub, logvar_loss_coef = state.apply_fn(
                params, Xbatch, train_mode=True
            )
            loss = PNNTrainer.pnn_loss(
                yhat, logvar, Ybatch, logvar_lb, logvar_ub, logvar_loss_coef
            )
            return loss, yhat

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, preds), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, (loss, preds)

    @staticmethod
    def pnn_loss(
        yhat, logvar, y, logvar_lb=None, logvar_ub=None, logvar_loss_coef=None
    ):
        sq_diffs = jnp.square(yhat - y)
        mse = jnp.mean(sq_diffs)
        loss = jnp.mean(sq_diffs * jnp.exp(-logvar) + logvar)
        if logvar_lb is not None:
            logvar_loss = jnp.abs(logvar_ub - logvar_lb) * logvar_loss_coef
            loss = loss + logvar_loss
        return loss

    def train(self, model, Xtrain, Ytrain, t, train_diag=None):
        tx = optax.adamw(self.lr, weight_decay=self.weight_decay)
        X = jnp.array(Xtrain)
        Y = jnp.array(Ytrain)
        if Y.ndim == 1:
            Y = Y[:, None]
        one_batch_X = X[: self.batch_size, ...]
        one_batch_Y = Y[: self.batch_size, ...]
        if self.params is not None:
            model_params = self.params[t]
            model_train_state = train_state.TrainState.create(
                apply_fn=model.apply, params=model_params, tx=tx
            )
            return model_train_state
        model_params = model.init(self.get_key(), one_batch_X)
        model_train_state = train_state.TrainState.create(
            apply_fn=model.apply, params=model_params, tx=tx
        )
        t = trange(self.num_iters)
        for step in t:
            model_train_state, mean_loss = self.train_epoch(
                model_train_state, Xtrain, Ytrain
            )
            t.set_postfix(loss=mean_loss)
        return model_train_state

    def train_epoch(self, train_state, X, Y):
        train_ds_size = X.shape[0]
        steps_per_epoch = train_ds_size // self.batch_size
        perms = jax.random.permutation(self.get_key(), train_ds_size)
        perms = perms[
            : steps_per_epoch * self.batch_size
        ]  # skip incomplete batch at end
        perms = perms.reshape((steps_per_epoch, self.batch_size))
        mean_loss = None
        t = tqdm(perms, disable=True)  # self.batch_size == X.shape[0])
        for perm in t:
            Xbatch = X[perm]
            Ybatch = Y[perm]
            train_state, (loss, preds) = self._train_step(train_state, Xbatch, Ybatch)
            if mean_loss is None:
                mean_loss = loss
            else:
                expweight = 0.5
                mean_loss = mean_loss * (1 - expweight) + loss * expweight
            t.set_postfix(loss=mean_loss)
        return train_state, mean_loss

    @staticmethod
    @jax.jit
    def _pred(X, *, train_state, **kwargs):
        return train_state.apply_fn(train_state.params, X, return_logvar=True)


class PECatboost:
    def __init__(self, model, num_samples, val_fraction=0.0, bootstrap=False, **kwargs):
        self.val_fraction = val_fraction
        self.num_samples = num_samples
        self.bootstrap = bootstrap
        self.models = []
        self.is_jax = False
        for seed in range(num_samples):
            new_model = deepcopy(model).set_params(random_seed=seed)
            self.models.append(new_model)

    def fit(self, X, Y):
        log.info(f"Fitting {self.num_samples} models...")
        X = np.array(X)
        Y = np.array(Y)
        for model in tqdm(self.models):
            if self.val_fraction > 0:
                Xtrain, Xval, Ytrain, Yval = train_test_split(
                    X, Y, test_size=self.val_fraction
                )
                model.fit(Xtrain, Ytrain, eval_set=(Xval, Yval))
            elif self.bootstrap:
                boot_idxes = np.random.randint(0, X.shape[0], size=X.shape[0])
                X_boot = X[boot_idxes]
                Y_boot = Y[boot_idxes]
                model.fit(X_boot, Y_boot)
            else:
                model.fit(X, Y)

    def predict(self, X):
        # TODO: convert this to mean and predictive variance
        X = np.array(X)
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        preds = np.array(preds)
        mean = np.mean(preds[..., 0], axis=0)
        var = np.var(preds[..., 0], axis=0)
        return mean, var

    def save_model(self):
        for i, model in enumerate(self.models):
            model_path = self.model_save_dir / f"model_{i}.joblib"
            joblib.dump(model, model_path)

    def load_model(self, path):
        self.models = []
        for i in range(self.num_samples):
            path = self.model_save_dir / f"model_{i}.joblib"
            self.models.append(joblib.load(path))


class CatBoostTrainer:
    def __init__(self, **kwargs):
        pass

    def train(self, model, Xtrain, Ytrain, t):
        model.fit(Xtrain, Ytrain)
