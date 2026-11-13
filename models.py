"""Five survival models behind one interface.

fit(X, time, event, X_val=None, time_val=None, event_val=None) -> self
predict_surv(X, grid) -> (n, len(grid)) survival probabilities
info -> dict of training details (epochs, device, extrapolation, ...)

Neural models use the supplied training-derived validation partition for early
stopping only; tree and Cox models train on the full training subset.
"""
import numpy as np

from common import eval_step_curves


def _stepfunctions_to_matrix(fns):
    times = np.asarray(fns[0].x, float)
    S = np.vstack([fn.a * np.asarray(fn.y, float) + fn.b for fn in fns])
    for fn in fns[1:4]:  # all functions from one model share a time axis
        if not np.array_equal(fn.x, fns[0].x):
            raise RuntimeError("step functions do not share time points")
    return times, S


class _SksurvModel:
    needs_validation = False

    def fit(self, X, time, event, **_):
        from sksurv.util import Surv
        self.model_ = self._make()
        self.model_.fit(X, Surv.from_arrays(event.astype(bool), time))
        self.train_max_time_ = float(time.max())
        self.info = {"train_max_time": self.train_max_time_}
        return self

    def predict_surv(self, X, grid):
        times, S = _stepfunctions_to_matrix(self.model_.predict_survival_function(X))
        self.info["grid_points_beyond_train_max"] = int(np.sum(grid > self.train_max_time_))
        return eval_step_curves(times, S, grid)


class CoxRidge(_SksurvModel):
    name = "cox_ridge"

    def __init__(self, alpha=1.0, seed=0, **_):
        self.alpha = alpha

    def _make(self):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        return CoxPHSurvivalAnalysis(alpha=self.alpha, ties="breslow", n_iter=200)


class RSF(_SksurvModel):
    name = "rsf"

    def __init__(self, n_estimators=300, min_samples_leaf=15, max_features="sqrt", n_jobs=1,
                 seed=0, **_):
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                       max_features=max_features, n_jobs=n_jobs, random_state=seed)

    def _make(self):
        from sksurv.ensemble import RandomSurvivalForest
        return RandomSurvivalForest(**self.kw)


class GBSA(_SksurvModel):
    name = "gbsa"

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
                 min_samples_leaf=10, seed=0, **_):
        self.kw = dict(loss="coxph", n_estimators=n_estimators, learning_rate=learning_rate,
                       max_depth=max_depth, subsample=subsample,
                       min_samples_leaf=min_samples_leaf, random_state=seed)

    def _make(self):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        return GradientBoostingSurvivalAnalysis(**self.kw)


class _PycoxModel:
    needs_validation = True

    def __init__(self, hidden=(32, 32), dropout=0.1, lr=1e-3, batch_size=256, max_epochs=500,
                 patience=20, device="cpu", seed=0, **kw):
        self.hidden, self.dropout, self.lr = list(hidden), dropout, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.device, self.seed, self.kw = device, seed, kw

    def _seed(self):
        import torch
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _train(self, model, x, y, x_val, y_val):
        """Warm-up epochs without stopping, then early stopping on the validation partition.

        The batch size gives at least 4 gradient steps per epoch on small subsets; without
        the warm-up, full-batch training on ~100 patients can stop at epoch 1 (pilot finding)."""
        import torchtuples as tt
        bs = int(min(self.batch_size, len(x), max(32, int(np.ceil(len(x) / 4)))))
        # Guarantee an observed event in every small-sample DeepSurv minibatch.
        if self.__class__.__name__ == "DeepSurv" and len(x) <= 100:
            censored = int(np.count_nonzero(np.asarray(y[1]) == 0))
            if censored == len(x):
                raise ValueError("DeepSurv fitting data contain no events")
            bs = next((b for b in range(bs, len(x) + 1)
                       if b > censored and
                       (len(x) % b == 0 or len(x) % b > censored)),
                      len(x))
        warmup = int(self.kw.get("min_epochs", 20))
        val = dict(verbose=False, val_data=(x_val, y_val), val_batch_size=len(x_val))
        if warmup:
            model.fit(x, y, bs, warmup, **val)
        log = model.fit(x, y, bs, self.max_epochs - warmup,
                        callbacks=[tt.callbacks.EarlyStopping(patience=self.patience)], **val)
        hist = log.to_pandas()  # cumulative over both calls
        after = hist["val_loss"].iloc[warmup:]
        self.info = {"epochs_run": int(len(hist)), "warmup_epochs": warmup,
                     "best_epoch": int(np.argmin(after.to_numpy())) + warmup + 1,
                     "best_val_loss": float(after.min()),
                     "device": str(self.device), "batch_size": bs}


class DeepSurv(_PycoxModel):
    name = "deepsurv"

    def fit(self, X, time, event, X_val=None, time_val=None, event_val=None):
        import torchtuples as tt
        from pycox.models import CoxPH
        self._seed()
        net = tt.practical.MLPVanilla(X.shape[1], self.hidden, 1, batch_norm=True,
                                      dropout=self.dropout, output_bias=False)
        self.model_ = CoxPH(net, tt.optim.Adam(self.lr), device=self.device)
        f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
        self._train(self.model_, f32(X), (f32(time), f32(event)),
                    f32(X_val), (f32(time_val), f32(event_val)))
        # Estimate baseline hazards on fitting data only. Validation outcomes may
        # guide early stopping but must not enter baseline estimation before
        # computing validation IBS for hyperparameter selection.
        x_fit = np.asarray(X, np.float32)
        t_fit = np.asarray(time, np.float32)
        e_fit = np.asarray(event, np.float32)
        self.model_.compute_baseline_hazards(input=x_fit, target=(t_fit, e_fit))
        self.train_max_time_ = float(t_fit.max())
        self.info["train_max_time"] = self.train_max_time_
        return self

    def predict_surv(self, X, grid):
        df = self.model_.predict_surv_df(np.asarray(X, np.float32))
        self.info["grid_points_beyond_train_max"] = int(np.sum(grid > self.train_max_time_))
        return eval_step_curves(df.index.to_numpy(float), df.to_numpy().T, grid)


class LogisticHazard(_PycoxModel):
    name = "logistic_hazard"

    def fit(self, X, time, event, X_val=None, time_val=None, event_val=None):
        import torchtuples as tt
        from pycox.models import LogisticHazard as LH
        self._seed()
        n_events = int(event.sum())  # fit-partition labels only; validation is held out for tuning
        k = int(min(self.kw.get("max_durations", 20),
                    max(self.kw.get("min_durations", 5),
                        n_events // self.kw.get("events_per_duration", 5))))
        # Cut points from training durations only.
        self.labtrans_ = LH.label_transform(k, scheme="quantiles")
        y = self.labtrans_.fit_transform(np.asarray(time, np.float32), np.asarray(event, np.float32))
        y_val = self.labtrans_.transform(np.asarray(time_val, np.float32),
                                         np.asarray(event_val, np.float32))
        net = tt.practical.MLPVanilla(X.shape[1], self.hidden, self.labtrans_.out_features,
                                      batch_norm=True, dropout=self.dropout)
        # Start from the training subset's marginal discrete hazards (logit scale), so early
        # stopping cannot return a network still at the uninformative 0.5-hazard initialisation.
        idx, ev = np.asarray(y[0]), np.asarray(y[1])
        at_risk = np.array([(idx >= j).sum() for j in range(self.labtrans_.out_features)])
        d = np.array([((idx == j) & (ev == 1)).sum() for j in range(self.labtrans_.out_features)])
        h = np.clip((d + 0.5) / (at_risk + 1.0), 1e-4, 1 - 1e-4)
        import torch
        with torch.no_grad():
            net.net[-1].bias.copy_(torch.as_tensor(np.log(h / (1 - h)), dtype=torch.float32))
        self.model_ = LH(net, tt.optim.Adam(self.lr), duration_index=self.labtrans_.cuts,
                         device=self.device)
        self._train(self.model_, np.asarray(X, np.float32), y,
                    np.asarray(X_val, np.float32), y_val)
        self.train_max_time_ = float(max(time.max(), time_val.max()))
        self.info.update({"n_durations_requested": k,
                          "n_cuts": int(len(self.labtrans_.cuts)),
                          "train_max_time": self.train_max_time_})
        return self

    def predict_surv(self, X, grid):
        df = self.model_.interpolate(10).predict_surv_df(np.asarray(X, np.float32))
        self.info["grid_points_beyond_train_max"] = int(np.sum(grid > self.train_max_time_))
        return eval_step_curves(df.index.to_numpy(float), df.to_numpy().T, grid)


MODELS = {"cox_ridge": CoxRidge, "rsf": RSF, "gbsa": GBSA,
          "deepsurv": DeepSurv, "logistic_hazard": LogisticHazard}


def make_model(name, params, seed, device="cpu", n_jobs=1):
    kw = dict(params)
    kw.update(seed=seed)
    if name == "rsf":
        kw["n_jobs"] = n_jobs
    if name in ("deepsurv", "logistic_hazard"):
        kw["device"] = device
    return MODELS[name](**kw)
