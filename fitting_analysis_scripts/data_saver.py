# -*- coding: utf-8 -*-
"""
data_saver.py - Result Persistence & Archiving Module

This module serves as the primary data output engine for the framework. 
It ensures that all mathematical models, diagnostic statistics, and 
cross-validation outputs are properly formatted, annotated with relevant 
metrological metadata, and safely persisted to the file system.

Capabilities:
-------------
1. Centralized Output Routing: Organizes all generated artifacts within a 
   master 'results' directory structure, maintaining project cleanliness.
2. Standardized Reporting: Exports comprehensive CSV files containing calibration 
   coefficients (ITS-90, Rational, Polynomial) and their associated uncertainties.
3. Diagnostic Archiving: Captures GoF metrics (AIC, BIC, Reduced Chi-Squared) 
   and residual sequences for downstream auditing and traceability.
"""

import pandas as pd
import numpy as np
import os
import logging
import fitting_analysis_scripts.function_defs as function_defs

# --- METADATA TRANSLATION CONSTANTS ---
TRANSFORMATION_MAP = {
    'raw_R': "x = R (Raw Resistance)",
    'W_TPW': "x = W = R / R(TPW)",
    'ln_W': "x = ln(W) = ln(R / R(TPW))",
    'W_Ne': "x = W_Ne = R / R(TPNe)",
    'ln_W_Ne': "x = ln(W_Ne) = ln(R / R(TPNe))",
    'W_Ar': "x = W_Ar = R / R(TPAr)",
    'ln_W_Ar': "x = ln(W_Ar) = ln(R / R(TPAr))",
    'ln_R': "x = ln(R)"
}

RATIONAL_NORM_MAP = {
    1: "x = (R-Rmin)/(Rmax-Rmin)", 
    2: "x = R/R_TPNe", 
    3: "x = R/R_TPH2O", 
    4: "x = R/R_TPAr",
    5: "x = (lnR-lnRmin)/(lnRmax-lnRmin)", 
    6: "x = ln(R/R_TPNe)", 
    7: "x = ln(R/R_TPH2O)", 
    8: "x = ln(R/R_TPAr)"
}


def get_global_results_path(relative_path: str) -> str:
    """
    Resolves and constructs an absolute path within the centralized 'results' 
    directory structure at the project root. Automatically creates the 
    target directory tree if it does not exist.
    """
    # 1. Locate the project root (up one level from 'fitting_analysis_scripts')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 2. Set the global results base
    global_results_base = os.path.join(project_root, 'results')
    
    # 3. Join the base with the provided relative path
    final_output_path = os.path.normpath(os.path.join(global_results_base, relative_path))
    
    # 4. Physically create the directory tree if it doesn't exist
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path, exist_ok=True)
        logging.info(f"Initialized output directory: {final_output_path}")
    
    return final_output_path

def save_its90_coeffs(coeffs: dict, output_path: str):
    """
    Exports calculated ITS-90 deviation coefficients to a dedicated CSV file.
    """
    filename = os.path.basename(output_path)
    
    # Redirect to global results folder under a subfolder 'ITS90'
    target_dir = get_global_results_path("ITS90_Calibration")
    final_save_path = os.path.join(target_dir, filename)

    df_coeffs = pd.DataFrame(list(coeffs.items()), columns=['Coefficient', 'Value'])
    df_coeffs.to_csv(final_save_path, sep=';', index=False, float_format='%.10e')
    logging.info(f"ITS-90 coefficients saved to: {final_save_path}")

def save_statistics(all_results: dict, data_label: str, num_points: int, file_base_name: str, output_dir: str):
    """
    Saves fitting statistics for all tested degrees to a single CSV file.

    The saved file includes standard goodness-of-fit metrics (R-squared, Chi-squared)
    and information criteria (AIC, BIC), as well as advanced diagnostic test
    results (Durbin-Watson, Breusch-Pagan).

    Args:
        all_results (dict): Dictionary where keys are degrees and values are result dicts.
        data_label (str): A descriptive label for the dataset (used internally).
        num_points (int): The number of data points in the analyzed set.
        file_base_name (str): The base name for the output file.
        output_dir (str): The directory where the file will be saved.
    """
    stats_data = []
    for complexity_level, result in all_results.items():
        row_data = {
            'complexity_level': complexity_level,
            'num_parameters': result['num_parameters'],
            'r_squared': result.get('r_squared'),
            'chi_squared': result.get('chi_squared'),
            'reduced_chi_squared': result.get('reduced_chi_squared'),
            'aic': result.get('aic'),
            'bic': result.get('bic'),
            'max_abs_stud_resid': result.get('max_abs_stud_resid'),
            'max_abs_residual_mK': result.get('max_abs_residual_mk'),
            'sum_of_absolute_residuals': result.get('sum_of_absolute_residuals'),
            'durbin_watson': result.get('durbin_watson'),
            'bp_lm_stat': result.get('bp_lm_stat'),
            'bp_p_value': result.get('bp_p_value')
        }
        stats_data.append(row_data)
    
    df_stats = pd.DataFrame(stats_data)
    
    if not df_stats.empty:
        df_stats.sort_values(by='complexity_level', inplace=True)
        if 'm' in next(iter(all_results.values())):
             df_stats.rename(columns={'complexity_level': 'm'}, inplace=True)

    output_filename = f"{file_base_name}_statistics.csv"
    target_dir = get_global_results_path(output_dir)
    output_path = os.path.join(target_dir, output_filename)
    df_stats.to_csv(output_path, sep=';', index=False)
    logging.info(f"Statistics saved to: {output_path}")


def save_parameters(all_results: dict, data_label: str, num_points: int, file_base_name: str, output_dir: str,
                    fitting_function_name: str, max_degree: int, B1_val: float, B2_val: float):
    """
    Exports fitted mathematical coefficients and their associated standard errors 
    across all evaluated complexities. Handles column padding for polynomials 
    and custom ordering for non-linear models like Sine waves.

    Args:
        all_results (dict): Dictionary containing results for all fitted degrees.
        data_label (str): A descriptive label for the dataset.
        num_points (int): The number of data points in the analyzed set.
        file_base_name (str): The base name for the output file.
        output_dir (str): The directory where the file will be saved.
        fitting_function_name (str): The name of the function used for fitting.
        max_degree (int): The maximum degree tested, used for column padding.
        B1_val (float): The min X value from the original dataset (for scaling).
        B2_val (float): The max X value from the original dataset (for scaling).
    """
    if not all_results:
        return

    func_info = function_defs.get_fitting_function(fitting_function_name)
    scaling_used = func_info and func_info.get('scaling_type', 'none') != 'none'
    
    records = []
    for degree, result in sorted(all_results.items()):
        row_data = {'degree': degree}
        if scaling_used:
            row_data['B1'], row_data['B2'] = B1_val, B2_val

        params = result['params']
        errors = result['param_errors']
        param_names = function_defs.get_param_names_for_function(fitting_function_name, len(params))
        
        for i, name in enumerate(param_names):
            row_data[name] = params[i]
            row_data[f"{name}_err"] = errors[i]
            
        records.append(row_data)

    df_params = pd.DataFrame.from_records(records)
    
    # --- Intelligent Column Ordering ---
    ordered_cols = ['degree']
    if scaling_used and 'B1' in df_params.columns: ordered_cols.extend(['B1', 'B2'])
    
    max_poly_params = max((d + 1 for d in all_results.keys() if isinstance(d, int)), default=0)
    for i in range(max_poly_params):
        p_name = f"A{i}"
        if p_name in df_params.columns: ordered_cols.extend([p_name, f"{p_name}_err"])
            
    if "Sine" in fitting_function_name:
        for name in ['Amplitude', 'Frequency', 'Phase']:
            if name in df_params.columns: ordered_cols.extend([name, f"{name}_err"])

    final_cols = ordered_cols + [c for c in df_params.columns if c not in ordered_cols]
    df_params = df_params[final_cols]

    output_filename = f"{file_base_name}_{num_points}pts_parameters.csv"
    target_dir = get_global_results_path(output_dir)
    output_path = os.path.join(target_dir, output_filename)
    df_params.to_csv(output_path, sep=';', index=False, float_format='%.8e')
    logging.info(f"Parameters saved to: {output_path}")


def save_best_fit_results(best_result: dict, data_label: str, num_points: int, file_base_name: str, output_dir: str, **kwargs):
    """
    Exports the fundamental curve data (measured points, fitted prediction line, 
    and standard/studentized residuals) for the single best-performing model.

    Args:
        best_result (dict): The dictionary of results for the best-fit model.
        data_label (str): A descriptive label for the dataset.
        num_points (int): The number of data points in the set.
        file_base_name (str): The base name for the output file.
        output_dir (str): The directory where the file will be saved.
        **kwargs: Catches unused arguments like B1_val and B2_val.
    """
    data_dict = {
        'x_transformed': best_result['x_raw_data'],
        'y_raw': best_result['y_data_data'],
        'y_fit': best_result['y_fit'],
        'residuals': best_result['residuals']
    }
    
    if 'x_untransformed_data' in best_result:
        data_dict['R_untransformed'] = best_result['x_untransformed_data']
    
    if best_result.get('studentized_residuals') is not None:
        data_dict['studentized_residuals'] = best_result['studentized_residuals']

    df_best_fit = pd.DataFrame(data_dict)
    
    cols_order = ['R_untransformed', 'x_transformed', 'y_raw', 'y_fit', 'residuals', 'studentized_residuals']
    final_cols = [col for col in cols_order if col in df_best_fit.columns]
    df_best_fit = df_best_fit[final_cols]
    
    output_filename = f"{file_base_name}_best_fit.csv"
    target_dir = get_global_results_path(output_dir)
    output_path = os.path.join(target_dir, output_filename)
    df_best_fit.to_csv(output_path, sep=';', index=False)
    logging.info(f"Best fit data saved to: {output_path}")
    
       
def save_outlier_variability_data(all_variability_results: dict, data_label_prefix: str,
                                  file_base_name: str, output_dir: str,
                                  B1_val: float, B2_val: float,
                                  fixed_polynomial_degree: int, sorted_num_removed_keys: list):
    """
    Saves combined statistics and parameters for the outlier variability test.

    This function creates two transposed summary files for easy comparison:
    1. Statistics File: Rows are statistic names (e.g., AIC), columns are for
       each step of outlier removal.
    2. Parameters File: Rows are parameter names (e.g., A0), columns are
       value/error pairs for each removal step.

    Args:
        all_variability_results (dict): Dict where keys are `num_removed` and values are results.
        data_label_prefix (str): Prefix for file names (e.g., "Outlier_Variability_Test").
        file_base_name (str): The base name of the original input file.
        output_dir (str): Directory where the files should be saved.
        B1_val (float): The min X value from the original dataset (for scaling).
        B2_val (float): The max X value from the original dataset (for scaling).
        fixed_polynomial_degree (int): The polynomial degree used for the test.
        sorted_num_removed_keys (list): Sorted list of `num_removed` keys for correct ordering.
    """
    if not all_variability_results:
        logging.warning("No results to save for outlier variability test.")
        return
    
    target_dir = get_global_results_path(output_dir)

   # --- 1. Prepare Transposed Statistics Matrix ---
    stats_df_data = {
        'Statistic': [
            'R_Squared', 'Chi_Squared', 'Reduced_Chi_Squared', 'AIC', 'BIC',
            'Sum_of_Absolute_Residuals', 'Num_Points_Remaining'
        ]
    }
    for num_removed_count in sorted_num_removed_keys:
        res = all_variability_results[num_removed_count]
        
        removed_index_info = "N/A"
        if res.get('removed_outlier_indices') and len(res['removed_outlier_indices']) >= num_removed_count:
            original_index_of_last_removed = res['removed_outlier_indices'][num_removed_count - 1]
            removed_index_info = f"{original_index_of_last_removed}"
        
        col_name = f'Removed_Point_Index_{removed_index_info}'
        stats_df_data[col_name] = [
            res.get('r_squared'), res.get('chi_squared'), res.get('reduced_chi_squared'),
            res.get('aic'), res.get('bic'), res.get('sum_of_absolute_residuals'),
            len(res.get('y_data_data', []))
        ]
    
    df_stats_combined = pd.DataFrame(stats_df_data)

    first_data_column_name = next((col for col in df_stats_combined.columns if col != 'Statistic'), 'Value_Column')
    df_stats_combined = pd.concat([
        df_stats_combined,
        pd.DataFrame([{'Statistic': 'B1 Value', first_data_column_name: B1_val}]),
        pd.DataFrame([{'Statistic': 'B2 Value', first_data_column_name: B2_val}])
    ], ignore_index=True)

    stats_file_name = f"{file_base_name}_{data_label_prefix.replace(' ', '_').lower()}_combined_statistics.csv"
    output_stats_path = os.path.join(target_dir, stats_file_name)
    df_stats_combined.to_csv(output_stats_path, sep=';', index=False, float_format='%.8e')
    logging.info(f"Saved combined outlier variability statistics to: {output_stats_path}")
    
    # --- 2. Prepare Transposed Parameters Matrix ---
    num_params = fixed_polynomial_degree + 1 
    param_names = [f'A{i}' for i in range(num_params)]

    params_combined_list = []
    for param_idx in range(num_params):
        param_row = {'Parameter_Name': param_names[param_idx]}
        for num_removed_count in sorted_num_removed_keys:
            res = all_variability_results[num_removed_count]
            
            removed_index_info = "N/A"
            if res.get('removed_outlier_indices') and len(res['removed_outlier_indices']) >= num_removed_count:
                original_index_of_last_removed = res['removed_outlier_indices'][num_removed_count - 1]
                removed_index_info = f"{original_index_of_last_removed}"
            
            col_val_name = f'Value_Removed_Index_{removed_index_info}'
            col_err_name = f'Error_Removed_Index_{removed_index_info}'

            if param_idx < len(res['params']):
                param_row[col_val_name] = res['params'][param_idx]
                param_row[col_err_name] = res['param_errors'][param_idx]
            else:
                param_row[col_val_name] = np.nan
                param_row[col_err_name] = np.nan
        params_combined_list.append(param_row)

    df_params_combined = pd.DataFrame(params_combined_list)
    
    df_params_combined = pd.concat([
        df_params_combined,
        pd.DataFrame([{'Parameter_Name': 'B1 Value', first_data_column_name: B1_val}]),
        pd.DataFrame([{'Parameter_Name': 'B2 Value', first_data_column_name: B2_val}])
    ], ignore_index=True)

    params_file_name = f"{file_base_name}_{data_label_prefix.replace(' ', '_').lower()}_combined_parameters.csv"
    output_params_path = os.path.join(target_dir, params_file_name) 
    df_params_combined.to_csv(output_params_path, sep=';', index=False, float_format='%.8e')
    logging.info(f"Saved combined outlier variability parameters to: {output_params_path}")

def _get_report_metadata(current_data, config, res=None):
    """
    Extracts operational metadata (Model Type, Transformation, R_Reference) 
    by recursively scanning configuration dictionaries. 
    Crucial for generating standardized report headers.
    
    Args:
        current_data (dict): The active dataset dictionary containing current session metadata.
        config (dict): The global configuration dictionary containing analysis parameters.
        res (dict, optional): The result dictionary from a specific model fit. Defaults to None.
    """
    current_data = current_data or {}
    config = config or {}
    res = res or {}
    
    # 1. Resolve Mathematical Model Name
    fit_name = config.get('analysis_params', {}).get('fitting_function_name')
    if not fit_name or str(fit_name) == "None":
        fit_name = res.get('fitting_function_name', "Model Result")
    
    # 2. Resolve X-Axis Transformation Protocol
    meta = current_data.get('x_transformation_metadata', config.get('x_transformation_metadata', {}))
    t_type = meta.get('type', 'raw_R')
    label = str(current_data.get('label', ''))
    for key in TRANSFORMATION_MAP.keys():
        if key != 'raw_R' and key in label:
            t_type = key
            break
    trans_label = TRANSFORMATION_MAP.get(t_type, t_type)

    # 3. Resolve Reference Resistance (R_ref)
    r_ref = meta.get('r_ref') or \
            config.get('analysis_params', {}).get('norm_params', {}).get('r_ref') or \
            config.get('analysis_params', {}).get('r_ref') or \
            current_data.get('r_ref')
            
    return fit_name, trans_label, r_ref

def _write_fit_core_logic(f, res, config, section_label, is_piecewise):
    """
    Core writing engine that structures the model boundaries, diagnostics, 
    and calculated parameters into an appendable CSV format.
    Dynamically adapts format for standard polynomials vs Rational functions.
    
    Args:
        f (file object): The open file handle where the CSV text will be written.
        res (dict): The result dictionary containing the fitted parameters and diagnostics.
        config (dict): The global configuration dictionary.
        section_label (str): The prefix applied to rows (e.g., 'GLOBAL', 'SEGMENT_1').
        is_piecewise (bool): Flag indicating if the 5-column piecewise format should be used.
    """
    prefix = f"{section_label}," if is_piecewise else ""

    # --- 1. Sub-Range Limits ---
    y_vals = res.get('y_data_data', res.get('y_raw', np.array([0])))
    r_vals = res.get('x_untransformed_data', res.get('R_untransformed', []))
    
    f.write(f"{prefix}--- LIMITS ---,-,-,-\n")
    f.write(f"{prefix}T_LIMIT_LOW,{y_vals.min():.4f},K,Min temperature\n")
    f.write(f"{prefix}T_LIMIT_HIGH,{y_vals.max():.4f},K,Max temperature\n")
    if len(r_vals) > 0:
        f.write(f"{prefix}R_MIN,{np.min(r_vals):.12e},Ohm,Min resistance\n")
        f.write(f"{prefix}R_MAX,{np.max(r_vals):.12e},Ohm,Max resistance\n")

    # --- 2. Statistical Diagnostics ---
    f.write(f"{prefix}--- DIAGNOSTICS ---,-,-,-\n")
    is_rational = 'n' in res or "Rational" in str(config.get('analysis_params', {}).get('fitting_function_name', ''))
    
    if is_rational:
        f.write(f"{prefix}MODEL_STRUCTURE,n={res.get('n')} m={res.get('m')},-,Rational order\n")
        choice = config.get('analysis_params', {}).get('norm_params', {}).get('choice')
        if choice in RATIONAL_NORM_MAP:
            f.write(f"{prefix}TRANS_METHOD,{RATIONAL_NORM_MAP[choice]},-,Scaling formula\n")
    else:
        deg = len(res.get('params', [])) - 1
        f.write(f"{prefix}POLYNOMIAL_DEGREE,{max(0, deg)},-,Polynomial degree\n")

    for k in ['reduced_chi_squared', 'aic', 'bic']:
        val = res.get(k, res.get('reduced_chi_sq' if k == 'reduced_chi_squared' else k, 'N/A'))
        f.write(f"{prefix}{k.upper()},{f'{val:.4f}' if isinstance(val, (float, int)) else 'N/A'},-,-\n")

    # --- 3. Parameter Coefficients ---
    f.write(f"{prefix}--- MODEL ---,-,-,-\n")
    params = res.get('params', [])
    errors = res.get('param_errors', [0]*len(params))
    
    if is_rational:
        n, m = int(res.get('n', 0)), int(res.get('m', 0))
        b0_zero = res.get('b0_is_zero', True)
        
        # Write Numerator components (N)
        for j in range(n + 1):
            if j < len(params):
                f.write(f"{prefix}N_{j},{params[j]:+18.12e},{errors[j]:.6e},Numerator\n")
                
        # Write Denominator components (M)
        offset = n + 1
        m_range = range(1, m + 1) if b0_zero else range(m + 1)
        for j in m_range:
            idx = offset + (j - 1 if b0_zero else j)
            if idx < len(params):
                f.write(f"{prefix}M_{j},{params[idx]:+18.12e},{errors[idx]:.6e},Denominator\n")
    else:
        # Standard polynomial coefficients (A)
        for j, p in enumerate(params):
            f.write(f"{prefix}A{j},{p:+18.12e},{errors[j]:.6e},Coefficient\n")

# =============================================================================
# --- PUBLIC API: COMPREHENSIVE REPORT GENERATION ---
# =============================================================================

def save_global_report(res, current_data, config, output_dir=None):
    """
    Generates a 4-column comprehensive CSV report containing metadata, limits, 
    diagnostics, and calculated coefficients for a Global (non-segmented) model.
    
    Args:
        res (dict): The result dictionary for the best-fit model.
        current_data (dict): The dictionary containing the raw dataset and session info.
        config (dict): The global configuration dictionary.
        output_dir (str, optional): An override path for saving the report. 
                                    Defaults to config['main_output_folder'].
    """
    if not res: return
    
    path = output_dir or config.get('main_output_folder')
    if not path: return
    
    target_dir = get_global_results_path(path)
    fit_name, trans_label, r_ref = _get_report_metadata(current_data, config, res)
    
    base_name = str(config.get('base_file_name', 'Dataset')).split('__')[0]
    num_pts = len(res.get('y_data_data', (current_data or {}).get('y', [])))
    file_path = os.path.join(target_dir, f"{base_name}_{num_pts}pts_comprehensive_report.csv")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("PARAMETER,VALUE,UNIT,COMMENT\n")
            f.write(f"Function_Type,{fit_name},-,Model type\n")
            f.write(f"X_Transformation,{trans_label},-,Representation\n")
            
            if r_ref is not None:
                f.write(f"R_Reference_Value,{float(r_ref):.8f},Ohm,Reference Resistance\n")
            
            f.write(",,,\n")
            _write_fit_core_logic(f, res, config, "GLOBAL", is_piecewise=False)
            
        logging.info(f"Report created with R_ref={r_ref}")
    except Exception as e:
        logging.error(f"Failed to save global report: {e}")

def save_piecewise_results(piecewise_results: list, current_data: dict, config: dict):
    """
    Generates a 5-column comprehensive CSV report for Segmented (Piecewise) models.
    Appends a 'SECTION' prefix to each row to differentiate between adjacent 
    topological segments.
    
    Args:
        piecewise_results (list): A list of result dictionaries, one for each segment.
        current_data (dict): The dictionary containing the raw dataset and session info.
        config (dict): The global configuration dictionary.
    """
    if not piecewise_results: return
    target_dir = get_global_results_path(config.get('main_output_folder', 'results'))
    file_path = os.path.join(target_dir, "piecewise_final_report.csv")

    fit_name, trans_label, r_ref = _get_report_metadata(current_data, config)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("SECTION,PARAMETER,VALUE,UNIT,COMMENT\n")
            f.write(f"METADATA,Function_Type,{fit_name},-,Type\n")
            f.write(f"METADATA,Num_Segments,{len(piecewise_results)},-,Segments\n")
            f.write(f"METADATA,X_Transformation,{trans_label},-,Data\n")
            if r_ref is not None:
                f.write(f"METADATA,R_Reference_Value,{float(r_ref):.8f},Ohm,Reference\n")
            
            for i, res in enumerate(piecewise_results):
                f.write(",\n")
                _write_fit_core_logic(f, res, config, f"SEGMENT_{i+1}", is_piecewise=True)
        logging.info(f"Piecewise report saved: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save piecewise report: {e}")