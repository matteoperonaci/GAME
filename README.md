# GAME: Genetic Algorithms with Marginalised Ensembles
GAME is a Python framework for the model-independent reconstruction of physical functions from data. It upgrades the standard Genetic Algorithms (GA) approach by introducing a marginalised ensemble methodology. Instead of selecting a single "best-fit" function, GAME computes a weighted average over an ensemble of reconstructions, significantly improving stability, smoothness, and the accuracy of derivatives.

This codebase is an advanced extension of the original Genetic Algorithms library by Savvas Nesseris, available at https://github.com/snesseris/Genetic-Algorithms.

## Key Features
- **Ensemble Averaging:** Mitigates the stochastic nature of GA by averaging over multiple runs and hyperparameter configurations.
    
- **Roughness Penalisation:** Uses a combined estimator $S = \chi^2 + \lambda R$ (where $R$ is curvature/roughness) to weigh models. This penalises unphysical oscillations while fitting the data.
    
- **L-Curve Optimisation:** Automatically selects the optimal regularisation parameter $\lambda$ (the trade-off between fit quality and smoothness) using the "elbow" method.
    
- **Robust Uncertainty Budget:**
    
    - **Statistical Error ($\delta f_{\rm PI}$):** Path-integral estimation of the error allowed by the data noise.
        
    - **Systematic Error ($\sigma_{\rm ens}$):** Weighted variance across the ensemble, capturing the model dependency and stability of the reconstruction.
        
- **Derivative Stability:** Specifically designed to fix the instability of derivatives in symbolic regression, making it ideal for reconstructing cosmological quantities like the dark energy equation of state $w(z)$.
