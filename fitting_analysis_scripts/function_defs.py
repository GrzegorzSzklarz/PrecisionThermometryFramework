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
- Kernels: Unified mathematical implementations (Polynomial, Hybrid, etc.) 
  to minimize redundancy.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval

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
        # Store the function and its metadata in the registry dictionary.
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
    Supports standard polynomials (A0, A1...), hybrid models, and static models.
    """
    func_info = get_fitting_function(function_name)
    if not func_info:
        return [f"Param{i}" for i in range(num_params or 0)]

    # Hybrid Models: N-3 Polynomial coeffs + 3 Sine coeffs
    if "Sine" in function_name:
        poly_count = (num_params - 3) if num_params and num_params >= 3 else 1
        poly_names = [f"A{i}" for i in range(poly_count)]
        return poly_names + ['Amplitude', 'Frequency', 'Phase']
    
    # Standard Polynomials: A0, A1, A2...
    if func_info["is_polynomial"]:
        return [f"A{i}" for i in range(num_params or 0)]
    
    # Static Parameter Models (e.g., Exponential)
    return func_info.get("param_names") or [f"Param{i}" for i in range(num_params or 0)]

# --- SHARED MATHEMATICAL KERNELS (DRY) ---

def _evaluate_polynomial(x, params):
    """
    Efficiently evaluates a polynomial in ascending order:
    Y = A0 + A1*x + A2*x^2 + ...
    """
    # np.polynomial.polynomial.polyval is faster and more stable for ascending coeffs
    return polyval(x, params)

def _evaluate_hybrid_sine(x, params):
    """Mathematical kernel for Polynomial + Sine wave models."""
    if len(params) < 4:
        # Minimum: A0 (offset) + Amp + Freq + Phase
        poly_params = [params[0]]
        sine_params = params[1:]
    else:
        poly_params = params[:-3]
        sine_params = params[-3:]
        
    amp, freq, phase = sine_params
    return _evaluate_polynomial(x, poly_params) + amp * np.sin(freq * x + phase)

# --- MODEL REGISTRATIONS ---

# 1. Standard Polynomials
# Note: Math is identical for all three; analyzer handles the 'x' preparation based on scaling_type.

@register_fitting_function("Polynomial N-th degree", scaling_type='none', is_polynomial=True)
def polynomial_standard(x, *params):
    return _evaluate_polynomial(x, params)

@register_fitting_function("Z-function (N-th degree polynomial)", scaling_type='linear', is_polynomial=True)
def polynomial_linear_scaled(x, *params):
    return _evaluate_polynomial(x, params)

@register_fitting_function("Log-scaled Z-function N-th degree", scaling_type='log', is_polynomial=True)
def polynomial_log_scaled(x, *params):
    return _evaluate_polynomial(x, params)

# 2. Hybrid Polynomial + Sine Models

@register_fitting_function("Polynomial N-th degree + Sine", scaling_type='none', is_polynomial=True)
def hybrid_sine_raw(x, *params):
    return _evaluate_hybrid_sine(x, params)

@register_fitting_function("Z-function (N-th degree polynomial) + Sine", scaling_type='linear', is_polynomial=True)
def hybrid_sine_linear_scaled(x, *params):
    return _evaluate_hybrid_sine(x, params)

@register_fitting_function("Log-scaled Z-function (N-th degree polynomial) + Sine", scaling_type='log', is_polynomial=True)
def hybrid_sine_log_scaled(x, *params):
    return _evaluate_hybrid_sine(x, params)

# 3. Static Parameter Models

@register_fitting_function("Exponential function", param_names=['A', 'k', 'C'])
def exponential_function(x, A, k, C):
    """Standard Exponential: Y = A * exp(k * x) + C"""
    return A * np.exp(k * x) + C

# 4. Special Workflows

@register_fitting_function("Rational Function", is_special_workflow=True)
def create_rational_function(n_degree, m_degree, b0_is_zero):
    """
    Factory function for Rational/Pade models.
    Structure: P(x, n) / Q(x, m)
    """
    def rational_func(x, *params):
        num_p = n_degree + 1
        p_coeffs = params[:num_p]
        h_coeffs = params[num_p:]
        
        numerator = _evaluate_polynomial(x, p_coeffs)
        
        # Denominator normalization: 1 + sum(h*x^l)
        # Shift powers if b0_is_zero is required
        offset = 1 if b0_is_zero else 0
        powers = np.arange(offset, len(h_coeffs) + offset)
        denominator = 1 + sum(h * (x**l) for l, h in zip(powers, h_coeffs))
        
        return numerator / denominator
    return rational_func