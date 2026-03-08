# -*- coding: utf-8 -*-
"""
outlier_analyzer.py

This module provides a robust suite of tools for identifying anomalies in 
calibration data and validating the fundamental assumptions of Ordinary 
Least Squares (OLS) regression.

Capabilities:
-------------
1. Outlier Identification: Implements Z-score, Interquartile Range (IQR), 
   and Externally Studentized Residuals methods to detect physical or 
   instrumental anomalies.
2. Residual Diagnostics: Validates homoscedasticity (Breusch-Pagan test) 
   and checks for residual autocorrelation (Durbin-Watson test).
3. Visual Validation: Generates Standard Residual and Normal Q-Q plots 
   to graphically assess the noise distribution.
"""

import os
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
import fitting_analysis_scripts.function_defs as function_defs
import fitting_analysis_scripts.plotter as plotter

# Default thresholds for outlier detection methods.
DEFAULT_Z_SCORE_THRESHOLD = 2.0
DEFAULT_IQR_FACTOR = 1.5  # Common values: 1.5 for mild, 3.0 for extreme outliers


def _build_design_matrix(x_data_for_fit: np.ndarray, fit_info: dict) -> np.ndarray:
    """
    Constructs the design matrix (X) required for statsmodels OLS approximations.
    Dynamically adjusts matrix dimensions based on the polynomial degree or model type.
    """
    func_name = fit_info.get("fitting_function_name")
    
    # Special case for Rational Function: approximate with its numerator polynomial
    if func_name == "Rational Function (Pade-like)":
        n_degree = fit_info.get('n', 1)
        return np.vander(x_data_for_fit, n_degree + 1, increasing=True)

    # Logic for standard functions
    func_info = function_defs.get_fitting_function(func_name)
    if func_info and func_info["is_polynomial"]:
        num_params = fit_info['num_parameters']
        if "Sine" in func_name:
            degree = num_params - 4
        else:
            degree = num_params - 1
        return np.vander(x_data_for_fit, degree + 1, increasing=True)
    else:
        # Fallback for simple non-polynomial models like 'linear'
        return sm.add_constant(x_data_for_fit)

def analyze_z_score(residuals: np.ndarray, temperatures, threshold: float = DEFAULT_Z_SCORE_THRESHOLD) -> np.ndarray:
    """
    Analyzes residuals using the Z-score method to identify outliers.

    An outlier is identified if its Z-score (the number of standard deviations
    it is from the mean) exceeds the specified threshold.

    """
    res = np.asarray(residuals)
    temp = np.asarray(temperatures)
        
    z_scores = np.abs((res - np.mean(res)) / np.std(res))
    outlier_indices = np.where(z_scores > threshold)[0]
    
    if len(outlier_indices) > 0:
        print(f"\n[!] Z-score method (threshold {threshold}) found {len(outlier_indices)} outliers:")
        print(f"{'Index':<8} | {'Temp [K]':<12} | {'Residual':<18}")
        for idx in outlier_indices:
            t_val = temp[idx] if idx < len(temp) else "N/A"
            r_val = res[idx]
            print(f"{idx:<8} | {t_val:<12.4f} | {r_val:<18.6f}")
    else:
        msg = "No outliers detected using the Z-score method."
        print(msg)
        logging.info(msg)
        
    return outlier_indices


def analyze_iqr(residuals: np.ndarray, temperatures, factor: float = DEFAULT_IQR_FACTOR) -> np.ndarray:
    """
    Analyzes residuals using the Interquartile Range (IQR) method.

    An outlier is identified if it falls below Q1 - (factor * IQR) or
    above Q3 + (factor * IQR)..
    """
    
    # Ensure input is a numpy array for statistical operations
    res = np.asarray(residuals)
    temp = np.asarray(temperatures)
    
    if len(res) == 0:
        return np.array([])

    Q1 = np.percentile(res, 25)
    Q3 = np.percentile(res, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)
    
    outlier_indices = np.where((res < lower_bound) | (res > upper_bound))[0]
    
    print(f"\n--- IQR Analysis (Factor = {factor}) ---")
    if len(outlier_indices) > 0:
        print(f"\n[!] IQR method (factor {factor}) found {len(outlier_indices)} outliers:")
        print(f"{'Index':<8} | {'Temp [K]':<12} | {'Residual':<18}")
        for idx in outlier_indices:
            t_val = temp[idx] if idx < len(temp) else "N/A"
            r_val = res[idx]
            print(f"{idx:<8} | {t_val:<12.4f} | {r_val:<18.6f}")
    else:
        msg = "No outliers found using the IQR method."
        print(msg)
        logging.info(msg)
        
    return outlier_indices


def analyze_studentized_residuals(x_raw, y_raw, best_fit_info, threshold=3.0):
    """
    Identifies outliers using externally studentized residuals.
    This is the most metrologically rigorous method as it accounts for 
    the varying leverage of data points across the independent variable domain.
    """
    print(f"\n--- Studentized Residuals Analysis (Threshold = {threshold}) ---")
    
    temp = np.asarray(y_raw)
    
    final_stud_res = []
    final_raw_res = []

    # --- Piecewise Fit Handling ---
    if isinstance(best_fit_info, list):
        for i, segment in enumerate(best_fit_info):
            seg_stud = list(segment.get('studentized_residuals', []))
            seg_raw = list(segment.get('residuals', []))
            
            if not seg_stud or not seg_raw:
                continue

            if i > 0:
                # Boundary Averaging: Smooth residuals at topological knots
                final_stud_res[-1] = (final_stud_res[-1] + seg_stud[0]) / 2.0
                final_raw_res[-1] = (final_raw_res[-1] + seg_raw[0]) / 2.0
                final_stud_res.extend(seg_stud[1:])
                final_raw_res.extend(seg_raw[1:])
            else:
                final_stud_res.extend(seg_stud)
                final_raw_res.extend(seg_raw)
        
        studentized_residuals = np.array(final_stud_res)
        raw_residuals = np.array(final_raw_res)
    
    # --- Case 2: Global Fit ---
    elif isinstance(best_fit_info, dict):
        studentized_residuals = best_fit_info.get('studentized_residuals')
        raw_residuals = best_fit_info.get('residuals')
    else:
        return np.array([])

    if studentized_residuals is None or len(studentized_residuals) == 0:
        print("[!] Error: No studentized residuals available.")
        return np.array([])

    # Detect outliers
    abs_stud_res = np.abs(studentized_residuals)
    outlier_indices = np.where(abs_stud_res > threshold)[0]

    if len(outlier_indices) > 0:
        print(f"\n[!] Studentized Residuals found {len(outlier_indices)} outliers:")
        print(f"{'Index':<8} | {'Temp [K]':<12} | {'Residual':<18} | {'Stud. Res.':<10}")

        for idx in outlier_indices:
            t_val = temp[idx] if idx < len(temp) else "N/A"
            r_val = raw_residuals[idx] if idx < len(raw_residuals) else 0.0
            s_val = studentized_residuals[idx] if idx < len(studentized_residuals) else 0.0
            
            print(f"{idx:<8} | {t_val:<12.4f} | {r_val:<18.6f} | {s_val:<12.3f}")
    else:
        print("[*] No outliers detected using the Studentized Residuals method.")
        
    return outlier_indices

def visualize_and_test_residuals(residuals: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray,
                                 best_fit_info: dict, output_dir: str, file_base_name: str,
                                 x_for_plot: np.ndarray = None):
    """
    Provides a comprehensive visualization and statistical testing of model residuals.

    This function generates a series of plots (Residuals, Q-Q, Studentized
    Residuals) and performs formal tests (Durbin-Watson, Breusch-Pagan) to
    diagnose the quality of the model fit. Each plot is shown on screen and then
    saved to a file. Test results are printed to the console for immediate review.
    """
    
    print("\n--- Residuals Visualization and Statistical Tests ---")
    num_points = len(x_raw)

    plotter.generate_diagnostic_plots(
        best_result=best_fit_info,
        output_dir=output_dir,
        file_base_name=file_base_name,
        num_points=num_points,
        interactive=True
    )
    
    studentized_residuals = best_fit_info.get('studentized_residuals')
    
    if studentized_residuals is not None:
        logging.info(f"Max Absolute Studentized Residual: {np.max(np.abs(studentized_residuals)):.4f}")
        logging.info(f"Points with |r| > 2: {np.sum(np.abs(studentized_residuals) > 2)}")
        logging.info(f"Points with |r| > 3: {np.sum(np.abs(studentized_residuals) > 3)}")

    # --- 4. Formal Statistical Tests ---
    # Durbin-Watson Test for Autocorrelation
    dw_test = durbin_watson(residuals)
    print(f"\n  Durbin-Watson test for autocorrelation of residuals: {dw_test:.4f}")
    print("  (Ideal value is ~2.0. Values < 1.5 often indicate positive autocorrelation)")

   # Breusch-Pagan Test for Heteroskedasticity
    try:
        X_for_sm = _build_design_matrix(x_raw, best_fit_info)
        if X_for_sm.shape[1] > 1 and len(residuals) > X_for_sm.shape[1]:
            bp_test = het_breuschpagan(residuals, X_for_sm)
            bp_msg = f"Breusch-Pagan Test -> LM Stat: {bp_test[0]:.4f} | P-value: {bp_test[1]:.4f}. "
            bp_msg += "Suggests Heteroskedasticity." if bp_test[1] < 0.05 else "No significant heteroskedasticity."
            logging.info(bp_msg)
        else:
            logging.info("Breusch-Pagan Test skipped: Insufficient degrees of freedom.")
    except Exception as e:
        logging.error(f"Breusch-Pagan Test computation failed: {e}")

    # --- 5. Export Diagnostic Data to CSV ---    
    try:
        diagnostic_data = {
            'x_raw': x_raw,
            'y_raw': y_raw,
            'y_fit': y_raw - residuals,
            'residuals_mK': residuals*1000,
        }
        if studentized_residuals is not None:
            diagnostic_data['studentized_residuals'] = studentized_residuals
            
        df_diag = pd.DataFrame(diagnostic_data)
        
        cols = ['x_raw', 'y_raw', 'y_fit', 'residuals_mK', 'studentized_residuals']
        final_cols = [c for c in cols if c in df_diag.columns]
        df_diag = df_diag[final_cols]

        csv_path = os.path.join(output_dir, f"{file_base_name}_{num_points}pts_diagnostic_data.csv")
        df_diag.to_csv(csv_path, sep=';', index=False, float_format='%.8f')
        logging.info(f"Diagnostic data saved to: {csv_path}")

    except Exception as e:
        logging.error(f"Could not save diagnostic data to CSV: {e}")
        
        