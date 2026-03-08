# -*- coding: utf-8 -*-
"""
analyzer.py

This module orchestrates the fitting process for standard mathematical models. 
It implements an automated, iterative complexity scan designed to identify 
the parsimonious model that best represents the physical behavior of the sensor 
without overfitting.

Key Methodologies:
------------------
1. Model Selection: Uses Information Criteria (AIC/BIC) to balance goodness-of-fit 
   against model complexity (number of parameters).
2. Numerical Stability: Implements X-axis scaling (Linear/Z-function or Log) 
   to prevent ill-conditioned matrices during high-degree polynomial fitting.
3. Diagnostic Rigor: Incorporates Durbin-Watson tests for autocorrelation 
   and Breusch-Pagan tests for heteroscedasticity.
4. Overfitting Protection: Enforces the N/3 rule, where the number of parameters k 
   is limited by the number of observations n.
"""

import math
import logging
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
import fitting_analysis_scripts.function_defs as function_defs
import fitting_analysis_scripts.plotter as plotter


def perform_analysis_and_save_results(data_label: str, y_data_set: np.ndarray, std_y_set: np.ndarray,
                                      x_raw_set: np.ndarray, std_x_set: np.ndarray,
                                      x_untransformed_set: np.ndarray,
                                      B1_val: float, B2_val: float, max_degree: int,
                                      file_base_name: str, output_dir: str,
                                      plotter_func: callable, saver_func_stats: callable,
                                      saver_func_params: callable, saver_func_best_fit: callable,
                                      fitting_function_name: str,
                                      transform_details: dict = None,
                                      fixed_degree: int = None,
                                      removed_outlier_indices_current_step: list = None):
    """
    Executes a complete numerical fitting session, including complexity scanning, 
    statistical validation, and automated reporting.

    The function iteratively evaluates models of increasing complexity. For each 
    iteration, it calculates standard errors and advanced diagnostic metrics. 
    The 'best' model is identified primarily through the Akaike Information 
    Criterion (AIC).
    
    Args:
        data_label (str): A descriptive label for the dataset being analyzed.
        y_data_set (np.ndarray): The dependent variable data (Y values).
        std_y_set (np.ndarray): The standard deviations of the Y values.
        x_raw_set (np.ndarray): The independent variable data for the fit (can be transformed, e.g., W).
        std_x_set (np.ndarray): The standard deviations of the X values.
        x_untransformed_set (np.ndarray): The original, untransformed X data (raw R), used for plotting.
        B1_val (float): The minimum X value from the original dataset, for scaling.
        B2_val (float): The maximum X value from the original dataset, for scaling.
        max_degree (int): The maximum polynomial degree to test.
        file_base_name (str): The base name for all output files.
        output_dir (str): The directory where results will be saved.
        plotter_func (callable): Function for plotting the main summary results.
        saver_func_stats (callable): Function for saving statistics.
        saver_func_params (callable): Function for saving fitted parameters.
        saver_func_best_fit (callable): Function for saving the best fit curve data.
        fitting_function_name (str): The name of the function to use for fitting.
        transform_details (dict, optional): Details of the physical transformation applied.
        fixed_degree (int, optional): If specified, only this single degree is fitted.
        removed_outlier_indices_current_step (list, optional): For tracking in variability tests.

    Returns:
        dict: A dictionary containing the results for all successfully fitted degrees, or None if analysis fails.
    """
    
    print(f"\n======== Analysis for: {data_label} ({len(y_data_set)} points) ========")

    if len(y_data_set) == 0:
        logging.warning(f"Skipping analysis for {data_label}: Dataset is empty.")
        return None
    
    # Recalculate local scaling boundaries to ensure [-1, 1] mapping if required
    B1_val = np.min(x_untransformed_set)
    B2_val = np.max(x_untransformed_set)

    # Resolve function metadata and pointer from registry
    func_info = function_defs.get_fitting_function(fitting_function_name)
    if not func_info:
        logging.error(f"Fitting function '{fitting_function_name}' not found. Skipping.")
        return None
    
    fitting_func = func_info["function"]
    scaling_type = func_info.get("scaling_type", "none")
    is_polynomial_function = func_info["is_polynomial"]

    # --- Numerical Pre-processing (X-Scaling) ---
    # Scaling improves the condition number of the Jacobian matrix.
    if scaling_type == 'linear':
        range_width = B2_val - B1_val
        x_data_for_fit = (2 * x_raw_set - B1_val - B2_val) / range_width if range_width != 0 else np.zeros_like(x_raw_set)
    elif scaling_type == 'log':
        if np.any(x_raw_set <= 0) or B1_val <= 0 or B2_val <= 0:
            logging.error(f"Logarithmic scaling failed for '{data_label}': R values must be positive. Skipping analysis.")
            return None
        ln_R = np.log(x_raw_set)
        ln_Rmin = np.log(B1_val)
        ln_Rmax = np.log(B2_val)
        denominator = ln_Rmax - ln_Rmin
        x_data_for_fit = (2 * ln_R - ln_Rmax - ln_Rmin) / denominator if denominator != 0 else np.zeros_like(x_raw_set)
    else:
        x_data_for_fit = x_raw_set

    results_for_current_data = {}
    sigma_for_fit_and_chi2 = std_y_set
    num_observations_set = len(y_data_set)
    
    # --- Dynamic Complexity Determination ---
    # Implements the N/3 heuristic to ensure the model does not exceed the degrees of freedom.
    degrees_to_fit = []
    if fixed_degree is not None:
        degrees_to_fit = [fixed_degree]
    elif is_polynomial_function:
        max_allowed_degree_by_N = math.ceil(num_observations_set / 3) + 1
        current_max_degree = min(max_degree, max_allowed_degree_by_N)
        logging.info(f"Max degree for this dataset is dynamically set to {current_max_degree} (based on N/3 rule and global max).")
        start_degree = 5 if current_max_degree >= 5 else 1
        if current_max_degree >= start_degree:
            degrees_to_fit = list(range(start_degree, current_max_degree + 1))
    
    if not degrees_to_fit:
        degrees_to_fit = [0] # Handle non-polynomial or static cases

    # --- Main Numerical Fitting Loop ---
    for degree_label in degrees_to_fit:
        try:
            # Generate starting heuristics for the optimizer
            if is_polynomial_function:
                num_poly_terms = degree_label if "Sine" not in fitting_function_name else degree_label + 1
                current_initial_guess = [np.mean(y_data_set)] + [0.1] * num_poly_terms
                if "Sine" in fitting_function_name:
                    current_initial_guess += [0,2, 0.1, 0.2] # Amplitude, Freq, Phase
            else:
                default_non_poly_initial_guesses = {"exponential": [1.0, -0.1, 1.0], "linear": [1.0, 0.0]}
                current_initial_guess = default_non_poly_initial_guesses.get(fitting_function_name, [])

            if not current_initial_guess or num_observations_set < len(current_initial_guess):
                continue

            # Non-linear Least Squares using Levenberg-Marquardt or TRF
            params, cov_matrix = curve_fit(
                f=fitting_func, xdata=x_data_for_fit, ydata=y_data_set,
                p0=current_initial_guess, sigma=sigma_for_fit_and_chi2, 
                absolute_sigma=True, maxfev=10000
            )
            
            # --- Goodness-of-Fit Computation ---
            y_fit_set = fitting_func(x_data_for_fit, *params)
            residuals_set = y_data_set - y_fit_set
            max_abs_residual_mk = np.max(np.abs(residuals_set)) * 1000
            perr = np.sqrt(np.diag(cov_matrix))
            n, k = num_observations_set, len(params)
            degrees_of_freedom = n - k
            
            # Information Criteria and Reduced Chi-Squared
            ss_residual = np.sum(residuals_set**2)
            ss_total = np.sum((y_data_set - np.mean(y_data_set))**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan
            chi_squared = np.sum((residuals_set / sigma_for_fit_and_chi2)**2)
            reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.nan
            aic = 2 * k + n * np.log(ss_residual / n) if ss_residual > 0 and n > 0 else np.inf
            bic = k * np.log(n) + n * np.log(ss_residual / n) if ss_residual > 0 and n > 0 else np.inf

            # --- Advanced Residual Diagnostics ---
            adv_stats = {'studentized_residuals': None}
            try:
                X_for_sm = np.vander(x_data_for_fit, degree_label + 1, increasing=True)
                if X_for_sm.shape[1] > 1 and n > X_for_sm.shape[1]:
                    ols_results = sm.OLS(y_data_set, X_for_sm).fit()
                    stud_res = ols_results.get_influence().resid_studentized_external
                    adv_stats['studentized_residuals'] = stud_res 
                    adv_stats['max_abs_stud_resid'] = np.max(np.abs(stud_res))
                    adv_stats['durbin_watson'] = durbin_watson(residuals_set)
                    bp_test = het_breuschpagan(residuals_set, X_for_sm)
                    adv_stats['bp_lm_stat'] = bp_test[0] 
                    adv_stats['bp_p_value'] = bp_test[1]
            except Exception as e:
                logging.warning(f"Could not compute advanced stats for degree {degree_label}: {e}")

            # Aggregate iteration results
            result_dict = {
                'degree': degree_label,
                'params': params, 'param_errors': perr, 'y_fit': y_fit_set, 'residuals': residuals_set,
                'r_squared': r_squared, 'chi_squared': chi_squared, 'reduced_chi_squared': reduced_chi_squared,
                'aic': aic, 'bic': bic, 'degrees_of_freedom': degrees_of_freedom, 'num_parameters': k,
                'sum_of_absolute_residuals': np.sum(np.abs(residuals_set)),
                'max_abs_residual_mk': max_abs_residual_mk,
                'x_raw_data': x_data_for_fit, 
                'x_untransformed_data': x_untransformed_set,
                'y_data_data': y_data_set, 'std_y_data': std_y_set, 'std_x_data': std_x_set,
                'removed_outlier_indices': removed_outlier_indices_current_step,
                'fitting_function_name': fitting_function_name
            }
            result_dict.update(adv_stats)
            results_for_current_data[degree_label] = result_dict
            
        except (RuntimeError, Exception) as e:
            logging.error(f"Fit failed for degree {degree_label}: {e}")
            continue
            
    if not results_for_current_data:
        print(f"No successful fits found for {data_label}.")
        return None

    # --- Process and Save Results ---

    file_base_with_pts = f"{file_base_name}_{num_observations_set}pts"

    # Step 1: Find best fit by AIC (primary criterion)
    best_degree_aic = min(results_for_current_data, key=lambda k: results_for_current_data[k]['aic'])
    best_result_aic = results_for_current_data[best_degree_aic]
    print(f"\n--- Best Fit (by AIC) was for Degree: {best_degree_aic} ---")

    # Step 2: Find best fit by Sum of Absolute Residuals and save its key results
    if 'sum_of_absolute_residuals' in next(iter(results_for_current_data.values())):
        best_degree_abs = min(results_for_current_data, key=lambda k: results_for_current_data[k]['sum_of_absolute_residuals'])
        best_result_abs = results_for_current_data[best_degree_abs]
        print(f"--- Best Fit (by Sum of Abs. Res.) was for Degree: {best_degree_abs} ---")

        file_base_for_abs = f"{file_base_with_pts}_best_abs"
        saver_func_best_fit(
            best_result=best_result_abs, data_label=data_label, 
            num_points=num_observations_set, file_base_name=file_base_for_abs, 
            output_dir=output_dir
        )
        plotter.generate_diagnostic_plots(
            best_result_abs, output_dir, file_base_for_abs, num_observations_set
        )

    # Invoke plotting engine
    plotter_func(
        best_result=best_result_aic, 
        data_label=data_label, 
        all_results_for_current_data=results_for_current_data, 
        num_points=num_observations_set, 
        output_dir=output_dir, 
        file_base_name=file_base_with_pts
    )
    
    # Save statistics
    saver_func_stats(
        results_for_current_data, 
        data_label, 
        num_observations_set, 
        file_base_with_pts, 
        output_dir
    )
    
    # Save parameters
    saver_func_params(
        all_results=results_for_current_data, 
        data_label=data_label, 
        num_points=num_observations_set,
        file_base_name=file_base_with_pts, 
        output_dir=output_dir, 
        fitting_function_name=fitting_function_name, 
        max_degree=max_degree, 
        B1_val=B1_val, 
        B2_val=B2_val
    )
    
    # Save best-fit specific data (AIC)
    saver_func_best_fit(
        best_result=best_result_aic, 
        data_label=data_label, 
        num_points=num_observations_set,
        file_base_name=file_base_with_pts,
        output_dir=output_dir
    )
    

    return results_for_current_data
