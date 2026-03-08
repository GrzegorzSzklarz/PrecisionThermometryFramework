"""
rational_function_handler.py

This module orchestrates the complete workflow for fitting metrological calibration 
data using Rational Functions (specifically, non-linear Padé-like approximants). 
It operates as a self-contained, high-performance execution unit triggered by main.py.

Workflow:
1. User defines a physical normalization protocol for the independent variable (Resistance).
2. User configures the denominator topology (specifically, the b_0 intercept term).
3. The engine deploys a parallelized 2D grid scan, evaluating all valid topological 
   combinations of numerator degree (n) and denominator degree (m).
4. For each (n, m) pair, the Levenberg-Marquardt solver optimizes the coefficients.
5. Extensive diagnostic reports and graphical summaries are generated for each numerator degree.
6. The global optimum is isolated using the Akaike Information Criterion (AIC).
7. Transitions into an interactive outlier rejection and cross-validation CLI loop.
"""

import os
import logging
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import fitting_analysis_scripts.logger_setup as logger_setup
import fitting_analysis_scripts.data_saver as data_saver
import fitting_analysis_scripts.plotter as plotter
import fitting_analysis_scripts.function_defs as function_defs
import interactive_handlers as handlers

def _worker_fit_task(args):
    """Parallel processing wrapper unpacking arguments for the core optimization solver."""
    x_data, y_data, std_y, n, m, b0_is_zero, x_untransformed, std_x_untransformed = args
    return _run_single_fit(x_data, y_data, std_y, n, m, b0_is_zero, x_untransformed, std_x_untransformed)

def _get_normalization_params(df: pd.DataFrame) -> dict:
    """
    Interactive CLI for defining the mathematical normalization strategy of the 
    measured resistance data before executing the rational regression.
    """
    print("\n--- Select Normalization for Rational Function ---")
    options = [
        "x = (R-Rmin) / (Rmax-Rmin)   [Linear Scaling]",
        "x = R / R_TPNe               [Ratio W_Ne]",
        "x = R / R_TPH2O              [Ratio W]",
        "x = R / R_TPAr               [Ratio W_Ar]",
        "x = (lnR-lnRmin)/(lnRmax-lnRmin) [Logarithmic Scaling]",
        "x = ln(R / R_TPNe)           [Natural Log of W_Ne]",
        "x = ln(R / R_TPH2O)          [Natural Log of W]",
        "x = ln(R / R_TPAr)           [Natural Log of W_Ar]"
    ]
    
    for i, opt in enumerate(options, 1): print(f"{i}. {opt}")
    while True:
        try:
            choice = int(input(f"Select an option (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options): break
            else: print("Error: Choice out of range.")
        except ValueError: print("Error: Invalid input.")
    
    label_map = {1: "x_scaled", 2: "WNe", 3: "W", 4: "WAr", 5: "ln_x_scaled", 6: "ln(WNe)", 7: "ln(W)", 8: "ln(WAr)"}
    params = {'choice': choice, 'label': label_map.get(choice, f"unknown_norm_{choice}")}

    if choice in [1, 5]:
        params['Rmin'], params['Rmax'] = df['R'].min(), df['R'].max()
    elif choice in [2, 3, 4, 6, 7, 8]:
        points = {2: "TPNe", 3: "TPH2O", 4: "TPAr", 6: "TPNe", 7: "TPH2O", 8: "TPAr"}
        params['r_ref'] = handlers.get_float_input(f"Enter resistance at {points[choice]} [Ω]: ")
    return params

def _apply_normalization(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Translates physical resistance values into the selected normalized mathematical domain."""
    choice = params['choice']
    r_vals = df['R'].values
    
    if choice == 1:
        rmin, rmax = params['Rmin'], params['Rmax']
        return (r_vals - rmin) / (rmax - rmin) if (rmax - rmin) != 0 else np.zeros_like(r_vals)
    elif choice == 5:
        rmin, rmax = params['Rmin'], params['Rmax']
        if np.any(r_vals <= 0) or rmin <= 0: raise ValueError("Values must be positive for log-scaling.")
        ln_r, ln_rmin, ln_rmax = np.log(r_vals), np.log(rmin), np.log(rmax)
        denominator = ln_rmax - ln_rmin
        return (ln_r - ln_rmin) / denominator if denominator != 0 else np.zeros_like(r_vals)
    elif choice in [2, 3, 4]:
        return r_vals / params['r_ref']
    elif choice in [6, 7, 8]:
        if np.any(r_vals <= 0) or params['r_ref'] <= 0: raise ValueError("Values must be positive.")
        return np.log(r_vals / params['r_ref'])
    return r_vals

def _ask_b0_choice() -> bool:
    """Queries user to determine if the denominator's constant term should be eliminated."""
    while True:
        choice = input("Fit the b_0 term in the denominator? (y/n, default n): ").strip().lower()
        if choice in ['y', 'n', '']:
            b0_is_zero = (choice != 'y')
            logging.info(f"Denominator b_0 term is zero: {b0_is_zero}")
            return b0_is_zero
        print("Invalid input. Please enter 'y' or 'n'.")

def _run_single_fit(x_data, y_data, std_y, n, m, b0_is_zero, x_untransformed, std_x_untransformed):
    """
    Executes the Levenberg-Marquardt optimization for a specific (n, m) topology.
    Calculates comprehensive statistical diagnostics (AIC, BIC, residuals) upon convergence.
    """
    f = function_defs.create_rational_function(n, m, b0_is_zero)
    num_params_p = n + 1
    num_params_h = m if b0_is_zero else m + 1
    total_params = num_params_p + num_params_h
    
    initial_guess = [np.mean(y_data)] + [0.0] * (total_params - 1)
    N = len(y_data)
    
    if N <= total_params:
        logging.warning(f"Skipping (n={n}, m={m}): Not enough data points ({N}) for {total_params} parameters.")
        return None

    try:
        params, cov_matrix = curve_fit(f, x_data, y_data, p0=initial_guess, sigma=std_y, absolute_sigma=True, maxfev=100000, method='lm')
        if np.isinf(cov_matrix).any(): 
            raise RuntimeError("Covariance matrix contains infinity")
        perr = np.sqrt(np.diag(cov_matrix))
        y_fit = f(x_data, *params)
        residuals = y_data - y_fit
        
        k, dof = len(params), N - len(params)
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((y_data - np.mean(y_data))**2)

        results = {
            'params': params, 
            'param_errors': perr, 
            'residuals': residuals, 
            'y_fit': y_fit,
            'r_squared': 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan,
            'chi_squared': np.sum((residuals / std_y)**2),
            'reduced_chi_squared': np.sum((residuals / std_y)**2) / dof if dof > 0 else np.nan,
            'aic': 2 * k + N * np.log(ss_residual / N) if ss_residual > 0 and N > 0 else np.inf,
            'bic': k * np.log(N) + N * np.log(ss_residual / N) if ss_residual > 0 and N > 0 else np.inf,
            'max_abs_residual_mk': np.max(np.abs(residuals)) * 1000,
            'sum_of_absolute_residuals': np.sum(np.abs(residuals)),
            'n': n, 'm': m, 'num_parameters': k, 
            'b0_is_zero': b0_is_zero,
            'x_raw_data': x_data,
            'x_untransformed_data': x_untransformed,
            'y_data_data': y_data,
            'std_y_data': std_y,
            'std_x_data': std_x_untransformed,
            'fitting_function_name': 'Rational Function (Pade-like)'
        }
        
        
        adv_stats = {'max_abs_stud_resid': np.nan, 
                     'durbin_watson': np.nan, 
                     'bp_lm_stat': np.nan, 
                     'bp_p_value': np.nan, 
                     'studentized_residuals': None}
        try:
            X_for_sm = np.vander(x_data, n + 1, increasing=True)
            if X_for_sm.shape[1] > 1 and N > X_for_sm.shape[1]:
                ols_results = sm.OLS(y_data, X_for_sm).fit()
                stud_res = ols_results.get_influence().resid_studentized_external
                adv_stats.update({
                    'studentized_residuals': stud_res,
                    'max_abs_stud_resid': np.max(np.abs(stud_res)),
                    'durbin_watson': durbin_watson(residuals),
                    'bp_lm_stat': het_breuschpagan(residuals, X_for_sm)[0],
                    'bp_p_value': het_breuschpagan(residuals, X_for_sm)[1]
                })
        except Exception: pass
        
        results.update(adv_stats)
        
        # Nomenclature generation for CSV export
        param_names = [f'p{i}' for i in range(n + 1)]
        if b0_is_zero:
            param_names += [f'h{i}' for i in range(1, m + 1)]
        else:
            param_names += [f'h{i}' for i in range(m + 1)]
        results['param_names'] = param_names
        
        return results
    except (RuntimeError, Exception) as e:
        logging.warning(f"Fit failed for (n={n}, m={m}): {e}")
        return None

def _save_rational_parameters(results_dict, num_points, file_base_name, output_dir):
    """Exports model coefficients across varying denominator complexities (m) for a fixed numerator (n)."""
    records = []
    max_p = max((res.get('n', 0) for res in results_dict.values()), default=0)
    max_m = max((len(res.get('params', [])) - (res.get('n', 0) + 1) for res in results_dict.values()), default=0)
    
    for m, result in sorted(results_dict.items()):
        row_data = {'m': m}
        for i, name in enumerate(result.get('param_names', [])):
            row_data[name] = result['params'][i]
            row_data[f"{name}_err"] = result['param_errors'][i]
        records.append(row_data)

    df_params = pd.DataFrame.from_records(records)
    if not df_params.empty: df_params.sort_values(by='m', inplace=True)
    
    ordered_cols = ['m']
    for i in range(max_p + 1): ordered_cols.extend([f"p{i}", f"p{i}_err"])
    for i in range(max_m + 1): ordered_cols.extend([f"h{i}", f"h{i}_err"])
    
    final_cols = [c for c in ordered_cols if c in df_params.columns]
    df_params = df_params[final_cols]

    output_path = os.path.join(output_dir, f"{file_base_name}_parameters.csv")
    df_params.to_csv(output_path, sep=';', index=False, float_format='%.8e')
    logging.info(f"Rational parameters saved to: {output_path}")

def _save_best_m_per_n_summary(all_fits, output_path, dataset_label, criterion='aic', filename_suffix=''):
    """Isolates and exports the optimal denominator degree (m) for every tested numerator degree (n)."""
    if not all_fits:
        return None
        
    summary_list = []
    for (n, m), result in all_fits.items():
        summary_list.append({
            'n': n,
            'm': m,
            'aic': result.get('aic', np.nan),
            'bic': result.get('bic', np.nan),
            'reduced_chi_squared': result.get('reduced_chi_squared', np.nan),
            'sum_of_absolute_residuals': result.get('sum_of_absolute_residuals', np.nan)
        })
    df = pd.DataFrame(summary_list)
    if df.empty:
        return None
        
    best_m_df = df.loc[df.groupby('n')[criterion].idxmin()]
    best_m_df = best_m_df.sort_values(by='n').reset_index(drop=True)
    
    filename = os.path.join(output_path, f"{dataset_label}_best_m_per_n{filename_suffix}.csv")
    best_m_df.to_csv(filename, sep=';', index=False, float_format='%.8f')
    
    logging.info(f"Summary of best 'm' per 'n' (criterion: {criterion}) saved to: {filename}")
    return best_m_df

def _save_full_statistics_summary(all_fits, output_path, dataset_label):
    """Exports a comprehensive matrix of GoF metrics for the entire 2D topology scan."""
    if not all_fits: return
    summary_list = []
    
    keys_to_extract = [
        'n', 'm', 'num_parameters', 'r_squared', 'chi_squared', 
        'reduced_chi_squared', 'aic', 'bic', 'max_abs_stud_resid',
        'max_abs_residual_mk', 'sum_of_absolute_residuals', 'durbin_watson', 
        'bp_lm_stat', 'bp_p_value'
    ]
    
    for (n, m), result in all_fits.items():
        summary_list.append({key: result.get(key) for key in keys_to_extract})
        
    df = pd.DataFrame(summary_list)
    df.sort_values(by=['n', 'm'], inplace=True)
    
    if 'max_abs_residual_mk' in df.columns:
        df.rename(columns={'max_abs_residual_mk': 'max_abs_residual_mK'}, inplace=True)

    final_cols_order = [
        'n', 'm', 'num_parameters', 'r_squared', 'chi_squared', 
        'reduced_chi_squared', 'aic', 'bic', 'max_abs_stud_resid',
        'max_abs_residual_mK', 'sum_of_absolute_residuals', 'durbin_watson', 
        'bp_lm_stat', 'bp_p_value'
    ]

    df_final = df[[col for col in final_cols_order if col in df.columns]]

    filename = os.path.join(output_path, f"{dataset_label}_full_scan_statistics.csv")
    df_final.to_csv(filename, sep=';', index=False, float_format='%.6f')
    logging.info(f"Full statistics summary saved to: {filename}")

def _plot_final_summary(best_overall_fit, best_per_n_df, x_norm, data_label, output_path, file_base, criterion_name="AIC"):
    """
    Generates a 3-panel metrological summary visualizing the topology optimization 
    landscape and the residual distribution of the global optimum.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(f'Rational Function Analysis Summary ({criterion_name} criterion): {data_label}', fontsize=16)

    # --- Panel 1: Global Optimum Residuals ---
    try:
        n_best, m_best = best_overall_fit['n'], best_overall_fit['m']
        # Do a sanity check if y_data_data is in the dict
        if 'y_data_data' not in best_overall_fit:
             raise ValueError("'y_data_data' not found in best_overall_fit dictionary")
        
        design_matrix = np.vander(x_norm, n_best + 1, increasing=True)
        ols_results = sm.OLS(best_overall_fit['y_data_data'], design_matrix).fit()
        stud_res = ols_results.get_influence().resid_studentized_external
        
        axes[0].plot(best_overall_fit['y_data_data'], stud_res, 'o', ms=4, alpha=0.7)
        axes[0].axhline(y=2, color='orange', ls=':', label='|Residuals| > 2')
        axes[0].axhline(y=-2, color='orange', ls=':')
        axes[0].axhline(y=3, color='red', ls=':', label='|Residuals| > 3')
        axes[0].axhline(y=0, color='black', linestyle='--')
        axes[0].set_title(f"Best Overall Fit (n={n_best}, m={m_best}): Studentized Residuals")
        axes[0].set_xlabel("Temperature, K")
        axes[0].set_ylabel("Studentized Residuals")
        axes[0].legend()
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Could not generate residuals plot:\n{e}", ha='center', va='center')
    axes[0].grid(True)

    n_vals = best_per_n_df['n']
    m_vals = best_per_n_df['m']

    # --- Panel 2: Physical Accuracy vs Topology ---
    ax2 = axes[1]
    y1_vals = best_per_n_df['sum_of_absolute_residuals']
    y2_vals = best_per_n_df['reduced_chi_squared']
    
    ax2_twin = ax2.twinx()
    p1, = ax2.plot(n_vals, y1_vals, 'o-', c='purple', label='Sum of Abs. Residuals')
    p2, = ax2_twin.plot(n_vals, y2_vals, 'x-', c='green', label='Reduced Chi-Squared')
    
    y1_min, y1_max = ax2.get_ylim()
    offset1 = (y1_max - y1_min) * 0.03  
    for n, m, val in zip(n_vals, m_vals, y1_vals):
        ax2.text(n, val + offset1, f"m={m}", fontsize=10, fontweight='bold', va='bottom', ha='center')

    ax2.set_ylabel('Sum of Absolute Residuals', color='purple')
    ax2_twin.set_ylabel('Reduced Chi-squared', color='green')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2.legend(handles=[p1, p2], loc='best')
    ax2.set_title('Best Fit Quality for Each Numerator Degree (n)')
    ax2.set_xlabel('Numerator Degree (n)')
    ax2.grid(True)
    
    # --- Panel 3: Information Criteria vs Topology ---
    ax3 = axes[2]
    y3_vals = best_per_n_df['aic']
    y4_vals = best_per_n_df['bic']
    
    ax3_twin = ax3.twinx()
    p3, = ax3.plot(n_vals, y3_vals, 's-', c='blue', label='AIC')
    p4, = ax3_twin.plot(n_vals, y4_vals, 'd-', c='red', label='BIC')
    
    y3_min, y3_max = ax3.get_ylim()
    offset3 = (y3_max - y3_min) * 0.03 
    for n, m, val in zip(n_vals, m_vals, y3_vals):
        ax3.text(n, val + offset3, f"m={m}", fontsize=10, fontweight='bold', va='bottom', ha='center')

    ax3.set_ylabel('AIC', color='blue')
    ax3_twin.set_ylabel('BIC', color='red')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.legend(handles=[p3, p4], loc='best')
    ax3.set_title('Information Criteria for Each Numerator Degree (n)')
    ax3.set_xlabel('Numerator Degree (n)')
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    try:
        suffix = "_abs_criterion" if "Abs" in criterion_name else "_aic_criterion"
        filename = f"{file_base}_summary_plots{suffix}.png"
        save_path = os.path.join(output_path, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Final summary plot ({criterion_name}) saved to: {save_path}")
    except Exception as e:
        logging.error(f"Could not save the final summary plot ({criterion_name}): {e}")

    plt.show(block=False)
    plt.close(fig)
    
def _run_full_rational_scan_for_subset(data_label, y_data_set, std_y_set, x_raw_set, std_x_set, x_untransformed_set, output_dir, **kwargs):
    """
    Executes a complete headless 2D (n, m) topological scan. 
    Crucial for Piecewise engines and automated sub-setting loops.
    """
    # --- 1. RECOVERING CONFIGURATION AND METADATA ---
    config_dict = kwargs.get('config_dict', kwargs.get('config')) or {}
    current_data_dict = kwargs.get('current_data_dict', kwargs.get('current_data')) or {}
    
    norm_params = kwargs.get('norm_params')
    b0_is_zero = kwargs.get('b0_is_zero')
    max_degree = kwargs.get('max_degree')
    
    num_points = len(y_data_set)
    print(f"\n--- Running full Rational Function scan for subset: {data_label} ({num_points} pts) ---")
    
    if x_raw_set is not None and len(x_raw_set) > 0:
        x_norm = x_raw_set
    else:
        subset_df = pd.DataFrame({'R': x_untransformed_set, 'T': y_data_set, 'Rstd': std_x_set, 'Tstd': std_y_set})
        x_norm = _apply_normalization(subset_df, norm_params)
    
    dynamic_max_degree_by_N = math.ceil(num_points / 3) + 1
    if max_degree is None:
        current_max_degree = int(dynamic_max_degree_by_N)
    else:
        current_max_degree = int(min(max_degree, dynamic_max_degree_by_N))

    # 2. Build tasks
    tasks = [(x_norm, y_data_set, std_y_set, n, m, b0_is_zero, x_untransformed_set, std_x_set) 
             for n in range(2, current_max_degree - 1) 
             for m in range(2, current_max_degree - n + 1)]
    
    if not tasks:
        print(f"[Warning] Not enough points in {data_label} for a scan.")
        return None

    # 3. Parallel Execution
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap_unordered(_worker_fit_task, tasks))
    
    all_fits = {(res['n'], res['m']): res for res in results if res}
    if not all_fits:
        print(f"[Warning] No successful fits for {data_label}.")
        return None

    # 4. Construct Hierarchical Output Directory
    results_by_n = {}
    for (n, m), res in all_fits.items():
        results_by_n.setdefault(n, {})[m] = res
        
    for n, results_for_n in sorted(results_by_n.items()):
        n_output_path = os.path.join(output_dir, f"n_{n}")
        os.makedirs(n_output_path, exist_ok=True)
        if not results_for_n: continue

        best_m = min(results_for_n, key=lambda k: results_for_n[k]['aic'])
        
        best_res = {
            **results_for_n[best_m], 
            'n': n, 
            'm': best_m,
            'x_raw_data': x_norm, 
            'x_untransformed_data': x_untransformed_set, 
            'y_data_data': y_data_set, 
            'std_y_data': std_y_set, 
            'std_x_data': std_x_set, 
            'fitting_function_name': 'Rational Function'
        }

        file_base = f"{data_label}_n{n}"
        
        data_saver.save_statistics(results_for_n, f"n={n}", num_points, file_base, n_output_path)
        _save_rational_parameters(results_for_n, num_points, file_base, n_output_path)
        data_saver.save_best_fit_results(best_res, f"n={n}, best_m={best_m}", num_points, file_base, n_output_path)
        
        plotter.plot_analysis_results(
            best_result=best_res, 
            data_label=f"{data_label}, n={n}", 
            all_results_for_current_data=results_for_n, 
            num_points=num_points, 
            output_dir=n_output_path, 
            file_base_name=file_base, 
            xlabel='Denominator Degree (m)', 
            degree_label="best m = ", 
            show_plot=False
        )

    # 5. Generate a Top-Level Metrological Summaries
    _save_full_statistics_summary(all_fits, output_dir, data_label)
    best_per_n_df_aic = _save_best_m_per_n_summary(all_fits, output_dir, data_label, 'aic', '_aic_criterion')
    
    overall_best_nm, overall_best_res = min(all_fits.items(), key=lambda item: item[1]['aic'])
    
    overall_best_full = {
        **overall_best_res, 
        'n': overall_best_nm[0],
        'm': overall_best_nm[1],
        'x_raw_data': x_norm, 
        'x_untransformed_data': x_untransformed_set, 
        'y_data_data': y_data_set, 
        'std_y_data': std_y_set, 
        'std_x_data': std_x_set, 
        'fitting_function_name': 'Rational Function'
    }
    
    _plot_final_summary(overall_best_full, best_per_n_df_aic, x_norm, data_label, output_dir, data_label, "AIC")
    data_saver.save_best_fit_results(overall_best_full, data_label, num_points, f"{data_label}_best_aic", output_dir)
    
    config_dict = kwargs.get('config_dict', kwargs.get('config'))
    current_data_dict = kwargs.get('current_data_dict', kwargs.get('current_data'))
    
    if config_dict:
        final_config = dict(config_dict)
        final_config['main_output_folder'] = output_dir
        data_saver.save_global_report(overall_best_full, current_data_dict, final_config)

    plotter.generate_diagnostic_plots(overall_best_full, output_dir, f"{data_label}_best_aic", num_points)
    
    return all_fits
    
def _run_fixed_rational_fit_for_variability(data_label, y_data_set, std_y_set, x_raw_set, std_x_set, x_untransformed_set, output_dir, **kwargs):
    """
    Executes a rigid regression for a single, pre-defined (n, m) topology. 
    Utilized exclusively by the sequential Outlier Variability testing suite.
    """
    n = kwargs.get('fixed_n')
    m = kwargs.get('fixed_m')
    b0_is_zero = kwargs.get('b0_is_zero')
    norm_params = kwargs.get('norm_params')
    
    subset_df = pd.DataFrame({'R': x_untransformed_set, 'T': y_data_set, 'Rstd': std_x_set, 'Tstd': std_y_set})
    x_norm = _apply_normalization(subset_df, norm_params)
    
    result = _run_single_fit(x_norm, y_data_set, std_y_set, n, m, b0_is_zero, x_untransformed_set, std_x_set)
    
    if result:
        num_points = len(y_data_set)
        result['fitting_function_name'] = f'Rational Function (n={n}, m={m})'
        
        res_dict = {f"n{n}_m{m}": result}
        data_saver.save_statistics(res_dict, data_label, num_points, data_label, output_dir)
        data_saver.save_best_fit_results(result, data_label, num_points, data_label, output_dir)
        
        plotter.plot_analysis_results(
            best_result=result, data_label=data_label, 
            all_results_for_current_data=res_dict, num_points=num_points, 
            output_dir=output_dir, file_base_name=data_label, 
            xlabel='Normalized X', degree_label="n+m=", show_plot=True
        )
        
def handle_rational_function_analysis(df_original, base_file_name_no_ext, data_folder, max_polynomial_degree):
    """
    Primary orchestrator for the Rational Function analysis framework.
    Manages user input, builds the topological grid, dispatches processing threads, 
    and handles the interactive cross-validation loops.
    """
    
    # Step 1: Initial user setup (normalization and b0 choice)
    try:
        norm_params = _get_normalization_params(df_original)
        norm_label = norm_params['label']
        b0_is_zero = _ask_b0_choice()
        if not b0_is_zero: norm_label += "_b0"
    except Exception as e:
        print(f"\nAn error occurred during setup: {e}"); return

    # Step 2: Setup output directories and logger
    rel_path = os.path.join(base_file_name_no_ext, "Rational_Function", norm_label)
    base_output_path = data_saver.get_global_results_path(rel_path)

   
    log_file_path = os.path.join(base_output_path, 'analysis_log.txt')
    logger_setup.setup_logger(log_file_path)
    logging.info(f"Logger initialized. Log file: {log_file_path}")
    logging.info(f"Normalization: {norm_label} with params: {norm_params}")
    logging.info(f"b_0 is zero: {b0_is_zero}")

    current_df = df_original.copy()
    
    # Step 3: Main analysis and outlier removal loop
    while True:
        num_points = len(current_df)
        dataset_label_with_pts = f"{base_file_name_no_ext}_{num_points}pts"
        
        if norm_params['choice'] in [1, 5]:
            old_min = norm_params.get('Rmin')
            old_max = norm_params.get('Rmax')
            norm_params['Rmin'] = current_df['R'].min()
            norm_params['Rmax'] = current_df['R'].max()
            
            if old_min != norm_params['Rmin'] or old_max != norm_params['Rmax']:
                logging.info(f"Boundary update detected: Rmin={norm_params['Rmin']}, Rmax={norm_params['Rmax']}")
        
        logging.info(f"--- Starting Analysis for {base_file_name_no_ext} ({num_points} points) ---")

        # Prepare data for the current run
        y_data, std_y = current_df['T'].values, current_df['Tstd'].values
        std_x_untransformed, x_current_untransformed = current_df['Rstd'].values, current_df['R'].values
        x_current_norm = _apply_normalization(current_df, norm_params)
        
        global_max_degree = max_polynomial_degree
        dynamic_max_degree_by_N = math.ceil(num_points / 3) + 1
        
        current_max_degree = int(min(global_max_degree, dynamic_max_degree_by_N)) 

        logging.info(f"Global max combined degree is {global_max_degree}. Based on N={num_points}, dynamic limit is {dynamic_max_degree_by_N}.")
        logging.info(f"Effective max combined degree for this scan is set to {current_max_degree}.")
        
        # Step 3a: Prepare tasks for parallel processing
        tasks = [(x_current_norm, y_data, std_y, n, m, b0_is_zero, x_current_untransformed, std_x_untransformed) 
                 for n in range(2, current_max_degree - 1) 
                 for m in range(2, current_max_degree - n + 1)]
        
        if not tasks:
            print("Warning: Not enough data points to generate any (n, m) combinations. Aborting analysis.")
            logging.warning(f"Scan skipped for N={num_points} and max_degree={current_max_degree}. No valid (n,m) combinations.")
            break

        print(f"\nStarting parallel fitting for {len(tasks)} combinations using {cpu_count()} CPU cores...")
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(_worker_fit_task, tasks), total=len(tasks), desc="Fitting Progress"))
        
        all_fits_this_run = {(res['n'], res['m']): res for res in results if res}

        # Step 3b: Process results and save intermediate reports for each `n`
        results_by_n = {}
        for (n, m), res in all_fits_this_run.items():
            results_by_n.setdefault(n, {})[m] = res
            
        print("\nScan complete. Saving intermediate results...")
        for n, results_for_current_n in sorted(results_by_n.items()):
            n_output_path = os.path.join(base_output_path, f"n_{n}")
            os.makedirs(n_output_path, exist_ok=True)
            logging.info(f"Processing and saving report for n={n}...")
            if not results_for_current_n: continue

            best_m = min(results_for_current_n, key=lambda k: results_for_current_n[k]['aic'])
            best_result_for_n = results_for_current_n[best_m]
            file_base_for_saving = f"{dataset_label_with_pts}_n{n}"
            
            best_result_for_plotter = {**best_result_for_n, 
                                       'x_raw_data': x_current_norm, 
                                       'x_untransformed_data': x_current_untransformed,
                                       'y_data_data': y_data, 
                                       'std_y_data': std_y, 
                                       'std_x_data': std_x_untransformed,
                                       'fitting_function_name': 'Rational Function (Pade-like)'}

            data_saver.save_statistics(results_for_current_n, f"n={n}", num_points, file_base_for_saving, n_output_path)         
            _save_rational_parameters(results_for_current_n, num_points,file_base_for_saving, n_output_path)          
            data_saver.save_best_fit_results(best_result_for_plotter, f"n={n}, best_m={best_m}", num_points, file_base_for_saving, n_output_path)
            
            plotter.plot_analysis_results(best_result=best_result_for_plotter, 
                                          data_label=f"{base_file_name_no_ext} ({num_points} pts), n={n}",
                                          all_results_for_current_data=results_for_current_n, 
                                          num_points=num_points,
                                          output_dir=n_output_path, 
                                          file_base_name=file_base_for_saving,
                                          xlabel='Denominator Degree (m)',
                                          degree_label="best m = ",
                                          show_plot=False)
        
        if not all_fits_this_run:
            print("No successful fits in the entire scan.")
            break
        
        # Step 3c: Generate and save final summary reports
        _save_full_statistics_summary(all_fits_this_run, base_output_path, dataset_label_with_pts)
        
        current_data_dict = {
            'y': y_data, 
            'std_y': std_y,
            'x': x_current_norm, 
            'std_x': std_x_untransformed,
            'x_untransformed': x_current_untransformed, 
            'label': dataset_label_with_pts, 'num_points': num_points
        }
        
        config_dict = {
            'base_file_name': base_file_name_no_ext,
            'main_output_folder': base_output_path,
            'analysis_params': {'norm_params': norm_params, 
                                'b0_is_zero': b0_is_zero, 
                                'fitting_function_name': 'Rational Function'}
        }
        
        # Report 1: AIC Criterion
        print("\n--- Generating Summary based on AIC ---")
        best_per_n_df_aic = _save_best_m_per_n_summary(all_fits_this_run, base_output_path, dataset_label_with_pts, 'aic', '_aic_criterion')
        overall_best_nm_aic, overall_best_result_aic = min(all_fits_this_run.items(), key=lambda item: item[1]['aic'])
        
        overall_best_result_full_aic = {**overall_best_result_aic,
                                        'x_raw_data': x_current_norm, 
                                        'x_untransformed_data': x_current_untransformed,
                                        'y_data_data': y_data, 
                                        'std_y_data': std_y, 
                                        'std_x_data': std_x_untransformed,
                                        'fitting_function_name': 'Rational Function (Pade-like)'}

        print(f"Overall Best Fit (by AIC): n={overall_best_nm_aic[0]}, m={overall_best_nm_aic[1]}")
        _plot_final_summary(overall_best_result_full_aic, best_per_n_df_aic, x_current_norm, f"{base_file_name_no_ext} ({num_points} pts)", base_output_path, dataset_label_with_pts, "AIC")
        data_saver.save_best_fit_results(overall_best_result_full_aic, dataset_label_with_pts, num_points, f"{dataset_label_with_pts}_best_aic", base_output_path)
        data_saver.save_global_report(overall_best_result_full_aic, current_data_dict, config_dict, output_dir=base_output_path)
        plotter.generate_diagnostic_plots(overall_best_result_full_aic, base_output_path, f"{dataset_label_with_pts}_best_aic", num_points)

        # Report 2: Sum of Absolute Residuals Criterion
        print("\n--- Generating Summary based on Sum of Absolute Residuals ---")
        best_per_n_df_abs = _save_best_m_per_n_summary(all_fits_this_run, base_output_path, dataset_label_with_pts, 'sum_of_absolute_residuals', '_abs_criterion')
        if best_per_n_df_abs is not None:
            overall_best_nm_abs, overall_best_result_abs = min(all_fits_this_run.items(), key=lambda item: item[1]['sum_of_absolute_residuals'])
            
            overall_best_result_full_abs = {**overall_best_result_abs, 
                                            'x_raw_data': x_current_norm, 'x_untransformed_data': x_current_untransformed,
                                            'y_data_data': y_data, 'std_y_data': std_y, 'std_x_data': std_x_untransformed,
                                            'fitting_function_name': 'Rational Function (Pade-like)'}
            
            print(f"Overall Best Fit (by Sum of Abs. Res.): n={overall_best_nm_abs[0]}, m={overall_best_nm_abs[1]}")
            _plot_final_summary(overall_best_result_full_abs, best_per_n_df_abs, x_current_norm, f"{base_file_name_no_ext} ({num_points} pts)", base_output_path, dataset_label_with_pts, "Sum of Absolute Residuals")
            data_saver.save_best_fit_results(overall_best_result_full_abs, dataset_label_with_pts, num_points, f"{dataset_label_with_pts}_best_abs", base_output_path)
            plotter.generate_diagnostic_plots(overall_best_result_full_abs, base_output_path, f"{dataset_label_with_pts}_best_abs", num_points)

        # Pack data for interactive handlers
        current_data_dict = {
            'y': y_data, 'std_y': std_y,
            'x': x_current_norm,
            'std_x': std_x_untransformed,
            'x_untransformed': x_current_untransformed, 
            'label': dataset_label_with_pts,
            'num_points': num_points
        }
        
        analysis_results_dict = {
            'full': all_fits_this_run, 
            'best_fit': overall_best_result_full_aic
        }
        
        # Prepare config dict for menu (Options 6, 7, 8 need this to be structured correctly)
        config_dict = {
            'base_file_name': base_file_name_no_ext,
            'main_output_folder': base_output_path,
            'mode_foldername': "Rational_Function",
            'data_folder': data_folder,
            'run_analysis_func': _run_full_rational_scan_for_subset,
            'is_special_workflow': True,
            'analysis_params': {
                'norm_params': norm_params,
                'b0_is_zero': b0_is_zero,
                'max_degree': global_max_degree  
            }
        }
        
        cleaned_data_dict, updated_results, should_exit, should_reanalyze = handlers.run_fit_analysis_loop(
            current_data=current_data_dict,
            analysis_results=analysis_results_dict,
            config=config_dict
        )
        
        # 1. Check for global exit ('q')
        if should_exit:
            print("\n[!] Program terminated by user.")
            sys.exit()
        
        # 2. Check for return to main menu ('0')
        if not should_reanalyze:
            return 
        
        # 3. Trigger re-analysis if data has been cleaned or explicitly requested
        if cleaned_data_dict['num_points'] < current_data_dict['num_points'] or should_reanalyze:
            current_df = pd.DataFrame({
                'R': cleaned_data_dict['x_untransformed'],
                'T': cleaned_data_dict['y'],
                'Rstd': cleaned_data_dict['std_x'],
                'Tstd': cleaned_data_dict['std_y']
            })
            continue
        else:
            break