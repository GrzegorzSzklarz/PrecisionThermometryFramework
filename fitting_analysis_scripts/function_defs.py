# -*- coding: utf-8 -*-
"""
function_defs.py - Mathematical Model Registry

This module provides a centralized repository for curve-fitting models used in 
thermometer calibration. It uses a decorator-based registration pattern to 
seamlessly integrate new models into the analysis workflow.

Architecture:
-------------
- Registry: A central dictionary stores function pointers and metadata.
- Decorator: '@register_fitting_function' handles metadata assignment.
- Kernels: Unified mathematical implementations (Polynomial, Chebyshev, Hybrid, Rational) 
  to minimize redundancy and ensure numerical stability.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.chebyshev import chebval

# A private dictionary that serves as the registry for all fitting functions.
_fitting_functions = {}

def register_fitting_function(name: str, scaling_type: str = 'none', is_polynomial: bool = False, param_names: list = None, is_special_workflow: bool = False):
    """
    Decorator to register a mathematical model with specific metadata.
    
    Args:
        name (str): Unique identifier for the model.
        scaling_type (str): Metadata for the analyzer ('none', 'linear', 'log').
        is_polynomial (bool): If True, complexity is determined by degree scan.
        param_names (list): Default labels for non-polynomial parameters.
        is_special_workflow (bool): If True, delegates to a custom handler (e.g., Rational).
    """
    def decorator(func):
        _fitting_functions[name] = {
            "function": func,
            "scaling_type": scaling_type,
            "is_polynomial": is_polynomial,
            "param_names": param_names,
            "is_special_workflow": is_special_workflow 
        }
        return func
    return decorator

# --- REGISTRY ACCESSORS ---

def get_fitting_function(name: str) -> dict:
    """Returns metadata and function pointer for the specified model name."""
    return _fitting_functions.get(name)

def list_fitting_functions() -> dict:
    """Returns a copy of the dictionary of all registered fitting functions."""
    return _fitting_functions.copy()

def get_param_names_for_function(function_name: str, num_params: int = None) -> list:
    """
    Generates descriptive parameter names for results reporting.
    Supports standard polynomials (A0, A1...), Chebyshev polynomials, hybrid models, and static models.
    """
    func_info = get_fitting_function(function_name)
    if not func_info:
        return [f"Param{i}" for i in range(num_params or 0)]

    # Hybrid Models: N-3 Polynomial coeffs + 3 Sine coeffs
    if "Sine" in function_name:
        poly_count = (num_params - 3) if num_params and num_params >= 3 else 1
        poly_names = [f"A{i}" for i in range(poly_count)]
        return poly_names + ['Amplitude', 'Frequency', 'Phase']
    
    # Standard & Chebyshev Polynomials: A0, A1, A2...
    if func_info["is_polynomial"]:
        return [f"A{i}" for i in range(num_params or 0)]
    
    # Static Parameter Models (e.g., Exponential)
    return func_info.get("param_names") or [f"Param{i}" for i in range(num_params or 0)]

# --- SHARED MATHEMATICAL KERNELS ---

def _evaluate_polynomial(x, params):
    """Evaluates a standard monomial polynomial series in ascending order."""
    return polyval(x, params)

def _evaluate_chebyshev(x, params):
    """
    Evaluates a Chebyshev polynomial series in ascending order:
    Y = C0*T0(x) + C1*T1(x) + C2*T2(x) + ...
    Expects x scaled to the domain [-1, 1].
    """
    return chebval(x, params)

def _evaluate_hybrid_sine(x, params):
    """Mathematical kernel for Monomial Polynomial + Sine wave models."""
    if len(params) < 4:
        poly_params = [params[0]]
        sine_params = params[1:]
    else:
        poly_params = params[:-3]
        sine_params = params[-3:]
        
    amp, freq, phase = sine_params
    return _evaluate_polynomial(x, poly_params) + amp * np.sin(freq * x + phase)

def _evaluate_hybrid_sine_chebyshev(x, params):
    """Mathematical kernel for Chebyshev Polynomial + Sine wave models."""
    if len(params) < 4:
        poly_params = [params[0]]
        sine_params = params[1:]
    else:
        poly_params = params[:-3]
        sine_params = params[-3:]
        
    amp, freq, phase = sine_params
    return _evaluate_chebyshev(x, poly_params) + amp * np.sin(freq * x + phase)

def _evaluate_chebyshev_rational(x, n_degree, m_degree, b0_is_zero, params):
    """
    Core kernel for Chebyshev Rational functions: P_n(x) / Q_m(x)
    Numerator and denominator are evaluated in Chebyshev basis T_i(x) on [-1, 1].
    """
    num_p = n_degree + 1
    p_coeffs = params[:num_p]
    h_coeffs = params[num_p:]
    
    # Numerator P_n(x) in Chebyshev basis
    numerator = _evaluate_chebyshev(x, p_coeffs)
    
    # Denominator Q_m(x) in Chebyshev basis: 1 + sum(h_i * T_{i+offset}(x))
    offset = 1 if b0_is_zero else 0
    denominator = np.ones_like(x, dtype=float)
    
    for idx, h in enumerate(h_coeffs):
        deg = idx + offset
        basis = [0.0] * (deg + 1)
        basis[deg] = 1.0
        denominator += h * chebval(x, basis)
        
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return numerator / denominator

# --- MODEL REGISTRATIONS ---

# =========================================================================
# 1. STANDARD MONOMIAL POLYNOMIALS
# =========================================================================

@register_fitting_function("Polynomial N-th degree", scaling_type='none', is_polynomial=True)
def polynomial_standard(x, *params):
    return _evaluate_polynomial(x, params)

@register_fitting_function("Z-function (N-th degree polynomial)", scaling_type='linear', is_polynomial=True)
def polynomial_linear_scaled(x, *params):
    return _evaluate_polynomial(x, params)

@register_fitting_function("Log-scaled Z-function N-th degree", scaling_type='log', is_polynomial=True)
def polynomial_log_scaled(x, *params):
    return _evaluate_polynomial(x, params)

# =========================================================================
# 2. HYBRID POLYNOMIAL + SINE MODELS
# =========================================================================

# @register_fitting_function("Polynomial N-th degree + Sine", scaling_type='none', is_polynomial=True)
# def hybrid_sine_raw(x, *params):
#     return _evaluate_hybrid_sine(x, params)

# @register_fitting_function("Z-function (N-th degree polynomial) + Sine", scaling_type='linear', is_polynomial=True)
# def hybrid_sine_linear_scaled(x, *params):
#     return _evaluate_hybrid_sine(x, params)

# @register_fitting_function("Log-scaled Z-function (N-th degree polynomial) + Sine", scaling_type='log', is_polynomial=True)
# def hybrid_sine_log_scaled(x, *params):
#     return _evaluate_hybrid_sine(x, params)

# =========================================================================
# 3. STATIC PARAMETER MODELS
# =========================================================================

# @register_fitting_function("Exponential function", param_names=['A', 'k', 'C'])
# def exponential_function(x, A, k, C):
#     """Standard Exponential: Y = A * exp(k * x) + C"""
#     return A * np.exp(k * x) + C

# =========================================================================
# 4. STANDARD MONOMIAL RATIONAL MODELS (INTERACTIVE NORMALIZATION)
# =========================================================================

@register_fitting_function("Rational Function", is_special_workflow=True)
def create_rational_function(n_degree, m_degree, b0_is_zero):
    """
    Factory function for Standard Rational/Padé models.
    Triggers the interactive normalization CLI menu in rational_function_handler.py.
    """
    def rational_func(x, *params):
        num_p = n_degree + 1
        p_coeffs = params[:num_p]
        h_coeffs = params[num_p:]
        
        numerator = _evaluate_polynomial(x, p_coeffs)
        offset = 1 if b0_is_zero else 0
        powers = np.arange(offset, len(h_coeffs) + offset)
        denominator = 1.0 + sum(h * (x**l) for l, h in zip(powers, h_coeffs))
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        
        return numerator / denominator
    return rational_func

# =========================================================================
# 5. CHEBYSHEV MODELS (POLYNOMIAL, HYBRID & AUTOMATIC RATIONALS)
# =========================================================================

# Standard & Log-scaled Chebyshev Polynomials
@register_fitting_function("Chebyshev N-th degree polynomial", scaling_type='linear', is_polynomial=True)
def polynomial_chebyshev_linear(x, *params):
    """Standard Chebyshev polynomial fit using linear [-1, 1] scaling."""
    return _evaluate_chebyshev(x, params)

@register_fitting_function("Log-scaled Chebyshev N-th degree", scaling_type='log', is_polynomial=True)
def polynomial_chebyshev_log(x_norm, *params):
    """Chebyshev polynomial fit on logarithmically scaled data mapped to [-1, 1]."""
    return _evaluate_chebyshev(x_norm, params)

# Hybrid Chebyshev Polynomial + Sine Models
# @register_fitting_function("Chebyshev N-th degree polynomial + Sine", scaling_type='linear', is_polynomial=True)
# def hybrid_sine_chebyshev_linear(x, *params):
#     """Hybrid Chebyshev polynomial + Sine model using linear [-1, 1] scaling."""
#     return _evaluate_hybrid_sine_chebyshev(x, params)

# @register_fitting_function("Log-scaled Chebyshev N-th degree + Sine", scaling_type='log', is_polynomial=True)
# def hybrid_sine_chebyshev_log(x_norm, *params):
#     """Hybrid Chebyshev polynomial + Sine model on logarithmically scaled data [-1, 1]."""
#     return _evaluate_hybrid_sine_chebyshev(x_norm, params)

# Chebyshev Rational Functions (Automatic Scaling, No interactive CLI menu)
@register_fitting_function("Chebyshev Rational Function", scaling_type='linear', is_special_workflow=True)
def create_chebyshev_rational_linear(n_degree, m_degree, b0_is_zero):
    """Factory function for Chebyshev Rational models using linear [-1, 1] scaling."""
    def chebyshev_rational_func(x, *params):
        return _evaluate_chebyshev_rational(x, n_degree, m_degree, b0_is_zero, params)
    return chebyshev_rational_func

@register_fitting_function("Log-scaled Chebyshev Rational Function", scaling_type='log', is_special_workflow=True)
def create_chebyshev_rational_log(n_degree, m_degree, b0_is_zero):
    """Factory function for Chebyshev Rational models using logarithmic [-1, 1] scaling."""
    def chebyshev_rational_func(x, *params):
        return _evaluate_chebyshev_rational(x, n_degree, m_degree, b0_is_zero, params)
    return chebyshev_rational_func