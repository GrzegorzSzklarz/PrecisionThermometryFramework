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
from fitting_analysis_scripts.analyzer import compute_fit_uncertainty_and_polynomial
from scipy.optimize._numdiff import approx_derivative

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
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    global_results_base = os.path.join(project_root, 'results')
    final_output_path = os.path.normpath(os.path.join(global_results_base, relative_path))
    
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path, exist_ok=True)
        logging.info(f"Initialized output directory: {final_output_path}")
    
    return final_output_path


def save_its90_coeffs(coeffs: dict, output_path: str):
    """
    Exports calculated ITS-90 deviation coefficients to a dedicated CSV file.
    """
    filename = os.path.basename(output_path)
    target_dir = get_global_results_path("ITS90_Calibration")
    final_save_path = os.path.join(target_dir, filename)

    df_coeffs = pd.DataFrame(list(coeffs.items()), columns=['Coefficient', 'Value'])
    df_coeffs.to_csv(final_save_path, sep=';', index=False, float_format='%.10e')
    logging.info(f"ITS-90 coefficients saved to: {final_save_path}")

    
def calculate_fit_uncertainty_vectorized(x_data, fitting_func, params, cov_matrix):
    """
    Calculates standard uncertainty of fit u(f(x)) at each point using the 
    covariance matrix (GUM error propagation: u^2 = J^T * Cov * J).
    """
    if cov_matrix is None or np.isinf(cov_matrix).any():
        return np.zeros_like(x_data)
        
    uncertainties = []
    for x_val in x_data:
        def model_wrapped(p):
            return fitting_func(x_val, *p)
        
        jacobian = approx_derivative(model_wrapped, params)
        variance = jacobian.T @ cov_matrix @ jacobian
        uncertainties.append(np.sqrt(max(0.0, variance)))
        
    return np.array(uncertainties)
    
def save_statistics(all_results: dict, data_label: str, num_points: int, file_base_name: str, output_dir: str):
    """
    Saves fitting statistics for all tested degrees to a single CSV file.
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
    
    save_jacobian_covariance_report(
        results_dict=all_results, 
        file_base_name=f"{file_base_name}_{num_points}pts", 
        output_dir=output_dir
    )


def save_parameters(all_results: dict, data_label: str, num_points: int, file_base_name: str, output_dir: str,
                    fitting_function_name: str, max_degree: int, B1_val: float, B2_val: float):
    """
    Exports fitted mathematical coefficients and their associated standard errors.
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
    Exports the fundamental curve data for the single best-performing model.
    """
    func_name = best_result.get('fitting_function_name')
    func_info = function_defs.get_fitting_function(func_name)
    fitting_func = func_info["function"] if func_info else None

    if 'u_fit_vector' in best_result and best_result['u_fit_vector'] is not None:
        u_fit_vector = best_result['u_fit_vector']
    elif fitting_func and 'cov_matrix' in best_result and best_result['cov_matrix'] is not None:
        u_fit_vector = calculate_fit_uncertainty_vectorized(
            best_result['x_raw_data'], 
            fitting_func, 
            best_result['params'], 
            best_result['cov_matrix']
        )
    else:
        u_fit_vector = np.zeros_like(best_result['x_raw_data'])
    
    best_result['u_fit_vector'] = u_fit_vector

    data_dict = {
        'x_transformed': best_result['x_raw_data'],
        'y_raw': best_result['y_data_data'],
        'y_fit': best_result['y_fit'],
        'U_fit_expanded_k2_mK': u_fit_vector * 2000.0,
        'residuals_mK': best_result['residuals'] * 1000.0
    }
    
    if 'x_untransformed_data' in best_result:
        data_dict['R_untransformed'] = best_result['x_untransformed_data']
    
    if best_result.get('studentized_residuals') is not None:
        data_dict['studentized_residuals'] = best_result['studentized_residuals']

    df_best_fit = pd.DataFrame(data_dict)
    
    cols_order = [
        'R_untransformed', 'x_transformed', 'y_raw', 'y_fit', 
        'U_fit_expanded_k2_mK', 'residuals_mK', 'studentized_residuals'
    ]
    
    final_cols = [col for col in cols_order if col in df_best_fit.columns]
    df_best_fit = df_best_fit[final_cols]
    
    output_filename = f"{file_base_name}_best_fit.csv"
    target_dir = get_global_results_path(output_dir)
    output_path = os.path.join(target_dir, output_filename)
    
    df_best_fit.to_csv(output_path, sep=';', index=False, float_format='%.8e')
    logging.info(f"Best fit dataset successfully exported to: {output_path}")    
    

def save_jacobian_covariance_report(results_dict: dict, file_base_name: str, output_dir: str):
    """
    Exports a dedicated metrological report containing full Covariance Matrices 
    and a 5th-degree polynomial approximation of the expanded fit uncertainty profile U_fit(T) [k=2, mK].
    """
    target_dir = get_global_results_path(output_dir)
    output_path = os.path.join(target_dir, f"{file_base_name}_jacobian_covariance.csv")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(";===================================================\n")
        f.write("; METROLOGICAL REPORT: JACOBIAN & COVARIANCE ENGINE\n")
        f.write("; Formulation: u_fit(x) = sqrt( J(x)^T * Cov * J(x) )\n")
        f.write(";===================================================\n\n")

        for key, result in sorted(results_dict.items()):
            func_name = result.get('fitting_function_name', 'Unknown Model')
            cov = result.get('cov_matrix')
            params = result.get('params', [])
            param_names = result.get('param_names', [f"P{i}" for i in range(len(params))])

            f.write(f";--- MODEL ENTRY: {key} ({func_name}) ---\n")
            f.write(f";Number of Parameters: {len(params)}\n")
            
            if cov is not None and not np.isinf(cov).any():
                f.write(";Parameter Labels:;" + ";".join(param_names) + "\n")
                f.write(";Parameter Values:;" + ";".join([f"{p:+18.12e}" for p in params]) + "\n")
                f.write(";Covariance Matrix:\n")
                
                f.write(";" + ";".join(param_names) + "\n")
                for row in cov:
                    f.write(";" + ";".join([f"{val:+.8e}" for val in row]) + "\n")
            else:
                f.write(";Covariance Matrix: N/A (Singular or ill-conditioned fit)\n")

            _, u_poly_coeffs, _, _ = compute_fit_uncertainty_and_polynomial(result, max_deg=5)

            if u_poly_coeffs is not None:
                f.write("\n;--- FIT UNCERTAINTY POLYNOMIAL MODEL U_fit(T) [mK] ---\n")
                f.write(";Formulation;U_fit(T) = u_0 + u_1*T + u_2*T^2 + u_3*T^3 + u_4*T^4 + u_5*T^5 [mK] (k=2)\n")
                f.write(";Coefficients_Order;u_0;u_1;u_2;u_3;u_4;u_5\n")
                
                coeffs_formatted = [f"{u_poly_coeffs[i]:+18.12e}" for i in range(6)]
                f.write(";Values;" + ";".join(coeffs_formatted) + "\n")
            else:
                f.write("\n;Fit Uncertainty Polynomial: Calculation unavailable\n")

            f.write("\n")

    logging.info(f"Jacobian & Covariance report with uncertainty polynomial saved to: {output_path}")


def _get_report_metadata(current_data, config, res=None):
    """
    Extracts operational metadata (Model Type, Transformation, R_Reference).
    """
    current_data = current_data or {}
    config = config or {}
    res = res or {}
    
    fit_name = config.get('analysis_params', {}).get('fitting_function_name')
    if not fit_name or str(fit_name) == "None":
        fit_name = res.get('fitting_function_name', "Model Result")
    
    meta = current_data.get('x_transformation_metadata', config.get('x_transformation_metadata', {}))
    t_type = meta.get('type', 'raw_R')
    label = str(current_data.get('label', ''))
    for key in TRANSFORMATION_MAP.keys():
        if key != 'raw_R' and key in label:
            t_type = key
            break
    trans_label = TRANSFORMATION_MAP.get(t_type, t_type)

    r_ref = meta.get('r_ref') or \
            config.get('analysis_params', {}).get('norm_params', {}).get('r_ref') or \
            config.get('analysis_params', {}).get('r_ref') or \
            current_data.get('r_ref')
            
    return fit_name, trans_label, r_ref


def _write_fit_core_logic(f, res, config, section_label, is_piecewise):
    """
    Core writing engine structuring boundaries, diagnostics, expanded fit uncertainties (k=2, mK),
    model parameters, full Jacobian Covariance Matrix, and a 5th-degree polynomial approximation of U_fit(T).
    """
    prefix = f"{section_label}," if is_piecewise else ""

    # --- 1. Sub-Range Limits ---
    y_vals = res.get('y_data_data', res.get('y_raw', np.array([0])))  # Physical Temperature T [K]
    r_vals = res.get('x_untransformed_data', res.get('R_untransformed', []))
    
    f.write(f"{prefix}--- LIMITS ---,-,-,-\n")
    f.write(f"{prefix}T_LIMIT_LOW,{np.min(y_vals):.4f},K,Min temperature\n")
    f.write(f"{prefix}T_LIMIT_HIGH,{np.max(y_vals):.4f},K,Max temperature\n")
    if len(r_vals) > 0:
        f.write(f"{prefix}R_MIN,{np.min(r_vals):.12e},Ohm,Min resistance\n")
        f.write(f"{prefix}R_MAX,{np.max(r_vals):.12e},Ohm,Max resistance\n")

    # --- 2. Diagnostics ---
    f.write("\n")
    f.write(f"{prefix}--- DIAGNOSTICS ---,-,-,-\n")
    is_rational = 'n' in res or "Rational" in str(config.get('analysis_params', {}).get('fitting_function_name', ''))
    
    if is_rational:
        f.write(f"{prefix}MODEL_STRUCTURE,n={res.get('n')} m={res.get('m')},-,Rational order P(x)/Q(x)\n")
    else:
        num_params_poly = len(res.get('params')) if res.get('params') is not None else 1
        deg = num_params_poly - 1
        f.write(f"{prefix}POLYNOMIAL_DEGREE,{max(0, deg)},-,Polynomial degree\n")

    for k in ['reduced_chi_squared', 'aic', 'bic']:
        val = res.get(k, res.get('reduced_chi_sq' if k == 'reduced_chi_squared' else k, 'N/A'))
        f.write(f"{prefix}{k.upper()},{f'{val:.4f}' if isinstance(val, (float, int)) else 'N/A'},-,-\n")

    # --- 3. Model Fit Uncertainty Metrics (GUM, k=2) ---
    cov_matrix = res.get('cov_matrix')
    params = res.get('params')

    u_nodes_mK, u_poly_coeffs, U_avg_k2_mK, U_max_k2_mK = compute_fit_uncertainty_and_polynomial(res, config, max_deg=5)

    if U_avg_k2_mK is not None:
        f.write(f"{prefix}FIT_UNCERTAINTY_EXP_AVG_K2_MK,{U_avg_k2_mK:.4f},mK,Mean expanded fit uncertainty U(T) (k=2)\n")
        f.write(f"{prefix}FIT_UNCERTAINTY_EXP_MAX_K2_MK,{U_max_k2_mK:.4f},mK,Max expanded fit uncertainty U(T) (k=2)\n")

    # --- 4. Model Coefficients (Excel Side-by-Side Layout) ---
    f.write("\n")
    f.write(f"{prefix}--- MODEL ---,-,-,-\n")
    
    param_errs = res.get('param_errors')
    num_params = len(params) if params is not None else 0
    errors = param_errs if param_errs is not None else [0.0] * num_params
    param_names = []
    
    if is_rational and params is not None:
        n = int(res.get('n', 0)) if res.get('n') is not None else 0
        m = int(res.get('m', 0)) if res.get('m') is not None else 0
        b0_zero = res.get('b0_is_zero', True)

        f.write(f"{prefix}Nominator,{n},Denominator,{m}\n")
        f.write(f"{prefix},value,,value\n")

        num_count = n + 1
        N_coeffs = params[:num_count]
        M_coeffs = params[num_count:]
        
        m_start_idx = 1 if b0_zero else 0
        max_rows = max(len(N_coeffs), len(M_coeffs))

        for r in range(max_rows):
            if r < len(N_coeffs):
                n_idx_str = f"{r}"
                n_val_str = f"{N_coeffs[r]:+18.12e}"
                param_names.append(f"N_{r}")
            else:
                n_idx_str = ""
                n_val_str = ""

            if r < len(M_coeffs):
                m_idx_val = r + m_start_idx
                m_idx_str = f"{m_idx_val}"
                m_val_str = f"{M_coeffs[r]:+18.12e}"
                param_names.append(f"M_{m_idx_val}")
            else:
                m_idx_str = ""
                m_val_str = ""

            f.write(f"{prefix}{n_idx_str},{n_val_str},{m_idx_str},{m_val_str}\n")

    elif params is not None:
        deg = len(params) - 1
        f.write(f"{prefix}POLYNOMIAL_DEGREE,{max(0, deg)},-,Polynomial degree\n")
        for j, p in enumerate(params):
            lbl = f"A{j}"
            param_names.append(lbl)
            err_val = errors[j] if j < len(errors) else 0.0
            f.write(f"{prefix}{lbl},{p:+18.12e},{err_val:.6e},Coefficient\n")

    # --- 5. Jacobian Covariance Matrix ---
    f.write("\n")
    f.write(f"{prefix}--- JACOBIAN COVARIANCE MATRIX ---,-,-,-\n")
    if cov_matrix is not None and not np.isinf(cov_matrix).any():
        f.write(f"{prefix}JACOBIAN_ENGINE,scipy.optimize._numdiff.approx_derivative,-,Numerical Jacobian\n")
        f.write(f"{prefix}UNCERTAINTY_FORMULA,u_fit(x) = sqrt( J(x)^T * Cov * J(x) ),-,GUM propagation\n")
        headers_str = ";".join(param_names) if param_names else ";".join([f"P{i}" for i in range(len(cov_matrix))])
        f.write(f"{prefix}COV_LABELS,{headers_str},-,Covariance matrix columns\n")
        for i, row in enumerate(cov_matrix):
            row_str = ";".join([f"{val:+.8e}" for val in row])
            lbl = param_names[i] if i < len(param_names) else f"ROW_{i}"
            f.write(f"{prefix}COV_ROW_{lbl},{row_str},-,Covariance matrix row\n")

    # --- 6. 5th-Degree Polynomial Approximation of Expanded Uncertainty U_fit(T) [mK] ---
    if u_poly_coeffs is not None:
        f.write("\n")
        f.write(f"{prefix}--- FIT UNCERTAINTY POLYNOMIAL MODEL U_fit(T) [mK] ---,-,-,-\n")
        f.write(f"{prefix}U_FIT_FORMULA,U_fit(T) = u_0 + u_1*T + u_2*T^2 + u_3*T^3 + u_4*T^4 + u_5*T^5 [mK],-,Expanded uncertainty approximation (k=2)\n")
        f.write(f"{prefix}U_FIT_COEFFS_ORDER,u_0;u_1;u_2;u_3;u_4;u_5,-,Order of polynomial coefficients\n")
        
        coeffs_str = ";".join([f"{u_poly_coeffs[idx]:+18.12e}" for idx in range(6)])
        f.write(f"{prefix}U_FIT_VALUES,{coeffs_str},mK,Polynomial coefficients u_0 to u_5\n")


# =============================================================================
# --- PUBLIC API: COMPREHENSIVE REPORT GENERATION ---
# =============================================================================

def save_global_report(res, current_data, config, output_dir=None):
    """
    Generates a 4-column comprehensive CSV report for a Global (non-segmented) model.
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
            
            f.write("\n")
            _write_fit_core_logic(f, res, config, "GLOBAL", is_piecewise=False)
            
        logging.info(f"Report created with R_ref={r_ref}")
    except Exception as e:
        logging.error(f"Failed to save global report: {e}")


def save_piecewise_results(piecewise_results: list, current_data: dict, config: dict):
    """
    Generates a 5-column comprehensive CSV report for Segmented (Piecewise) models.
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
                f.write("\n")
                _write_fit_core_logic(f, res, config, f"SEGMENT_{i+1}", is_piecewise=True)
        logging.info(f"Piecewise report saved: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save piecewise report: {e}")
        
def save_stitched_dataset_to_csv(piecewise_results: list, config: dict):
    """
    Compiles the constrained mathematical vectors from all distinct segments 
    into a single, continuous DataFrame and exports it to a system CSV report,
    including expanded (k=2) fit uncertainties in mK.
    """
    base_dir = config.get('main_output_folder', 'results')
    model_name = config.get('analysis_params', {}).get('model_type', 'Piecewise_Model')
    
    all_data = []
    
    for i, res in enumerate(piecewise_results):
        t_meas = res.get('y_data_data')
        r_meas = res.get('x_untransformed_data')
        t_fit = res.get('y_fit')
        res_k = res.get('residuals')
        u_fit_vec = res.get('u_fit_vector')  # Standard uncertainty vector [K] (k=1)
        
        if t_meas is None or r_meas is None or t_fit is None:
            continue
            
        res_mk = res_k * 1000.0
        
        # Calculate expanded uncertainty U_fit in mK (k=2)
        if u_fit_vec is not None and len(u_fit_vec) == len(t_meas):
            U_k2_mK = np.array(u_fit_vec) * 2000.0  # k=2, K -> mK
        else:
            U_k2_mK = np.full(len(t_meas), np.nan)
        
        for idx in range(len(t_meas)):
            all_data.append({
                'Segment': i + 1,
                'T_measured_K': t_meas[idx],
                'R_measured_Ohm': r_meas[idx],
                'T_fitted_K': t_fit[idx],
                'U_fit_expanded_k2_mK': U_k2_mK[idx],
                'Residual_mK': res_mk[idx]
            })
            
    if all_data:
        df = pd.DataFrame(all_data)
        df.sort_values(by='T_measured_K', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        out_path = os.path.join(base_dir, f"{model_name}_stitched_full_data.csv")
        df.to_csv(out_path, sep=';', index=False, float_format="%.8f")
        logging.info(f"Stitched dataset with uncertainty exported to: {out_path}")