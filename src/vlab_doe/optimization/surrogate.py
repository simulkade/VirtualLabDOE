"""Phase 4 — Gaussian-Process surrogate model (botorch / GPyTorch).

Wraps a ``botorch.models.SingleTaskGP`` with a clean ``fit`` / ``predict``
interface that handles all the normalisation, tensor conversion, and
un-normalisation transparently.

Design notes:
- Inputs are normalised to [0, 1] per factor using the training data bounds.
- Outputs are standardised (zero mean, unit variance) for numerical stability.
- The GP is fitted by maximising the exact marginal log-likelihood (MLL) via
  L-BFGS-B (botorch's ``fit_gpytorch_mll``).
- ``predict`` returns the posterior mean and standard deviation in the
  *original* output scale so callers never need to think about normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood


@dataclass
class GPSurrogate:
    """A fitted GP surrogate with posterior prediction.

    Parameters
    ----------
    factors:
        Input factors defining the domain and physical bounds.
    response_name:
        Name of the modelled response column in the training DataFrame.
    """

    factors: Sequence[object]   # Sequence[Factor] — avoid circular import
    response_name: str
    # Internal state set by fit(); not user-supplied
    _model: SingleTaskGP | None = field(default=None, repr=False, compare=False)
    _x_bounds: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _y_mean: float = field(default=0.0, repr=False, compare=False)
    _y_std: float = field(default=1.0, repr=False, compare=False)

    def fit(self, data: pd.DataFrame) -> "GPSurrogate":
        """Fit the GP to a design + response table.

        Parameters
        ----------
        data:
            DataFrame with columns for each factor (by name) and the response.
            All rows are used as training data.
        """
        factor_names = [f.name for f in self.factors]
        X = torch.tensor(
            data[factor_names].values, dtype=torch.double
        )
        Y = torch.tensor(
            data[[self.response_name]].values, dtype=torch.double
        )

        # Physical bounds tensor  shape (2, n_factors)
        bounds = torch.tensor(
            [[f.low for f in self.factors], [f.high for f in self.factors]],
            dtype=torch.double,
        )
        self._x_bounds = bounds

        # Normalise inputs to [0, 1]
        X_norm = normalize(X, bounds)

        # Standardise outputs
        self._y_mean = float(Y.mean())
        self._y_std = float(Y.std()) if float(Y.std()) > 1e-12 else 1.0
        Y_std = (Y - self._y_mean) / self._y_std

        model = SingleTaskGP(X_norm, Y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        self._model = model
        return self

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior ``(mean, std)`` at query points *x*.

        Parameters
        ----------
        x:
            DataFrame with columns for each factor.

        Returns
        -------
        mean:
            Posterior mean, shape ``(n_points,)``, in original response units.
        std:
            Posterior standard deviation, same shape and units.
        """
        if self._model is None:
            raise RuntimeError("GPSurrogate.predict called before fit().")

        factor_names = [f.name for f in self.factors]
        X = torch.tensor(x[factor_names].values, dtype=torch.double)
        X_norm = normalize(X, self._x_bounds)

        self._model.eval()
        with torch.no_grad():
            posterior = self._model.posterior(X_norm)
            mean_std = posterior.mean.squeeze(-1).numpy()
            var_std = posterior.variance.squeeze(-1).numpy()

        mean = mean_std * self._y_std + self._y_mean
        std = np.sqrt(np.clip(var_std, 0.0, None)) * self._y_std
        return mean, std
