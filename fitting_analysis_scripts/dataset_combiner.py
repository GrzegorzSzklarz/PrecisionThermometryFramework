# -*- coding: utf-8 -*-
"""
dataset_combiner.py

This module orchestrates the division of calibration data into discrete thermal 
sub-ranges (segments) and manages the mathematical boundaries (knots) between them.

Key Metrological Features:
--------------------------
1. Dataset Fusion: Merges multiple calibration runs while suppressing duplicate 
   measurements at the exact same physical coordinates.
2. C0 Continuity Enforcement: Applies constrained least-squares regression to 
   ensure adjacent polynomial segments intersect perfectly at the defined knots 
   (zero-order continuity).
3. Scale-Agnostic Anchoring: Dynamically translates physical boundary coordinates 
   (e.g., Ohms) into the specific mathematical domain of the model (e.g., Log(W), 
   linear mapping to [-1, 1]) before applying constraints.
4. Knot Optimization: Micro-adjusts the resistance anchor point at the boundary 
   to minimize the combined Residual Sum of Squares (RSS) of adjacent segments, 
   mimicking the physical behavior of the sensor.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
from scipy.optimize import OptimizeWarning
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
import fitting_analysis_scripts.data_loader as data_loader

# =============================================================================
# --- [SECTION 1: DATA MANIPULATION & TOPOLOGY SETUP] ---
# =============================================================================

def combine_with_secondary_dataset(current_data: dict, config: dict) -> dict:
    """
    Interactively merges the active dataset with an external CSV file.
    Automatically detects and drops duplicate measurements to prevent 
    statistical weighting bias in overlapping temperature ranges.
    """
    print("\n--- Step 1: Select Second Dataset to Combine ---")
    default_folder = config.get('data_folder', 'data')
    csv_files = glob.glob(os.path.join(default_folder, '*.csv'))
    
    second_file_path = None
    while True:
        print(f"\nFiles in ({default_folder}):")
        print("  0. [Enter custom path manually]")
        for i, filepath in enumerate(csv_files, start=1):
            print(f"  {i}. {os.path.basename(filepath)}")
            
        try:
            choice = int(input("\nSelect file: ").strip() or -1)
            if choice == 0:
                path = input("Path: ").strip().strip("\"'")
                if os.path.isfile(path): second_file_path = path 
                break
            elif 1 <= choice <= len(csv_files):
                second_file_path = csv_files[choice - 1] 
                break
        except ValueError: print("Invalid input.")
        
    if not second_file_path:
        return current_data

    try:
        df_second, _ = data_loader.load_data(second_file_path)
        df_curr = pd.DataFrame({'T': current_data['y'], 'Tstd': current_data['std_y'],
                                'R': current_data['x_untransformed'], 'Rstd': current_data['std_x']})
        
        # Concatenate and strip physical duplicates
        df_comb = pd.concat([df_curr, df_second], ignore_index=True)
        df_comb.drop_duplicates(subset=['T', 'R'], keep='first', inplace=True)
        df_comb.sort_values(by='T', inplace=True)
        
        current_data.update({'y': df_comb['T'].values, 'std_y': df_comb['Tstd'].values,
                             'x': df_comb['R'].values, 'std_x': df_comb['Rstd'].values,
                             'x_untransformed': df_comb['R'].values.copy(),
                             'num_points': len(df_comb), 'piecewise_mode': 'combined'})
        return _configure_piecewise_params(current_data)
    
    except Exception as e:
        logging.error(f"Dataset fusion failed during execution: {e}", exc_info=True)
        return current_data

def prepare_piecewise_division(current_data: dict, config: dict) -> dict:
    """Activates the piecewise (segmented) analytical workflow for a single dataset."""
    current_data['piecewise_mode'] = 'divided'
    return _configure_piecewise_params(current_data)

def _configure_piecewise_params(current_data: dict) -> dict:
    """
    Interactive CLI for defining topological knots (segment boundaries).
    Snaps user-defined temperature inputs to the closest actual measured 
    data point to avoid interpolation errors at the boundaries.
    """
    try: 
        num_funcs = int(input("How many segments? (default 2): ") or 2)
    except ValueError:
        num_funcs = 2

    if num_funcs <= 1: 
        current_data['piecewise_mode'] = 'none'
        return current_data

    splits = []
    min_t, max_t = current_data['y'].min(), current_data['y'].max()
    
    for i in range(1, num_funcs):
        while True:
            try:
                t_in_str = input(f" Define boundary #{i} [K] ({min_t:.1f} - {max_t:.1f}): ").replace(',', '.').strip()
                if not t_in_str: continue
                
                t_in = float(t_in_str)
                if min_t < t_in < max_t:
                    # Snap to the nearest actual measurement point
                    idx = (np.abs(current_data['y'] - t_in)).argmin()
                    snapped_t = current_data['y'][idx]
                    splits.append(snapped_t)
                    print(f"   -> Snapped to nearest actual data point: {snapped_t:.3f} K")
                    break
                else:
                    print(f" [Error] Boundary must be strictly between {min_t:.1f} and {max_t:.1f} K.")
            except ValueError: 
                print(" [Error] Invalid numerical input.")
                
    splits.sort()
    current_data.update({'piecewise_num_funcs': num_funcs, 'piecewise_splits_T': splits})
    return current_data

def get_math_x_from_phys_r(r_phys, metadata):
    """
    Translates a physical resistance value (Ohms) into the specific 
    mathematical domain required by the current fitting model 
    (e.g., W ratio, natural logarithm of W).
    """
    trans_type = metadata.get('type', 'raw_R')
    r_ref = metadata.get('r_ref', 1.0)
    
    if trans_type in ['W_TPW', 'W_Ne', 'W_Ar']:
        return r_phys / r_ref
    elif trans_type in ['ln_W', 'ln_W_Ne', 'ln_W_Ar']:
        return np.log(r_phys / r_ref) if r_phys > 0 else 0
    elif trans_type == 'ln_R':
        return np.log(r_phys) if r_phys > 0 else 0
    return r_phys

# =============================================================================
# --- [SECTION 2: CORE PIECEWISE ENGINE] ---
# =============================================================================

def run_intelligent_piecewise_analysis(current_data: dict, config: dict) -> tuple:
    """
    Executes the piecewise fitting sequence.
    
    Workflow:
    1. Segments data based on user-defined boundaries.
    2. Performs an unconstrained optimal fit for each segment.
    3. Iteratively applies C0 topological constraints to adjacent segments.
    4. Bypasses constraint logic for Rational Functions (which are handled natively).
    """
    
    splits = current_data.get('piecewise_splits_T', [])
    num_funcs = current_data.get('piecewise_num_funcs', 1)
    boundaries = [current_data['y'].min()] + splits + [current_data['y'].max()]
    
    results = []
    all_stats_list = []
    run_analysis_func = config['run_analysis_func']
    base_dir = config['main_output_folder']

    # Identify if the active model is a Rational Function
    is_rational = config.get('is_special_workflow', False) or \
                  "Rational" in config.get('analysis_params', {}).get('fitting_function_name', '')

    for i in range(num_funcs):
        t_start, t_end = boundaries[i], boundaries[i+1]
        mask = (current_data['y'] >= t_start) & (current_data['y'] <= t_end)
        
        seg_dir = os.path.join(base_dir, f"Segment_{i+1}")
        os.makedirs(seg_dir, exist_ok=True)
        
        segment_params = config['analysis_params'].copy()
        segment_params['show_plots'] = False
        
        # --- Explicit Local Normalization Handling ---
        # Crucial for Rational functions where domain scaling must be bounded 
        # specifically to the active segment's extrema, not the global dataset.
        r_seg = current_data['x_untransformed'][mask]
        
        if is_rational:
            norm_method = segment_params.get('norm_params', {}).get('choice')
        else:
            norm_method = segment_params.get('normalization_method')

        if is_rational and norm_method in [1, 5]:
            logging.info(f"Segment {i+1}: Explicitly recalculating local X for Rational Normalization {norm_method}.")
            
            if norm_method == 1:
                r_min, r_max = np.min(r_seg), np.max(r_seg)
                if r_max > r_min:
                    x_input_set = (r_seg - r_min) / (r_max - r_min)
                else:
                    x_input_set = np.zeros_like(r_seg)
            elif norm_method == 5:
                ln_r = np.log(r_seg)
                ln_r_min, ln_r_max = np.min(ln_r), np.max(ln_r)
                if ln_r_max > ln_r_min:
                    x_input_set = (ln_r - ln_r_min) / (ln_r_max - ln_r_min)
                else:
                    x_input_set = np.zeros_like(ln_r)
        else:
            # Standard polynomials use globally defined physical transformations
            x_input_set = current_data['x'][mask]
        
        # 1. Generate unconstrained optimal fit for the segment
        seg_dict = run_analysis_func(
            data_label=f"Seg_{i+1}", 
            y_data_set=current_data['y'][mask], 
            x_raw_set=x_input_set, 
            x_untransformed_set=r_seg, 
            std_y_set=current_data['std_y'][mask], 
            std_x_set=current_data['std_x'][mask],
            output_dir=seg_dir, 
            **segment_params  
        )
        
        if not seg_dict: 
            logging.error(f"Fitting failed for Segment_{i+1}")
            continue

        # Select the best model complexity via Akaike Information Criterion (AIC)
        best_key = min(seg_dict, key=lambda k: seg_dict[k]['aic'])
        current_res = seg_dict[best_key]

        # --- 2. C0 Continuity Enforcement (Smart Knot Logic) ---
        if i == 0:
            # First segment serves as the foundational anchor
            results.append(current_res)
        else:
            if is_rational:
                logging.info(f"Rational Function detected. Bypassing knot optimization for segment {i+1}.")
                current_res['is_constrained'] = False
                current_res['fit_status'] = 'Independent Optimal (Rational)'
                results.append(current_res)
            else:
                prev_seg = results[i-1]
                t_split = boundaries[i]
                try:
                    # Anchor the current segment to the tail of the previous segment
                    fit_prev, fit_curr, _ = find_optimal_knot_with_fallback(prev_seg, current_res, t_split, current_data)
                    results[i-1] = fit_prev
                    results.append(fit_curr)
                except Exception as e:
                    logging.warning(f"Knot optimization failed at {t_split}K: {e}. Using independent fit.")
                    results.append(current_res)
            
        all_stats_list.append(seg_dict)

    if results:
        from fitting_analysis_scripts.dataset_combiner import save_stitched_dataset_to_csv
        save_stitched_dataset_to_csv(results, config)

    return results, all_stats_list

def run_constrained_fit(best_fit_info: dict, t_anchor: float, r_phys_anchor: float, current_data: dict) -> dict:
    """
    Executes a constrained least-squares polynomial regression.
    Forces the resulting polynomial to pass exactly through the coordinate (t_anchor, r_phys_anchor).
    
    If the solver cannot converge or produces a physically invalid result (>100mK error), 
    it attempts a rigid Delta Shift (Y-intercept offset), and ultimately reverts to 
    an unconstrained fit if all topological enforcements fail.
    """
    trans_meta = current_data.get('x_transformation_metadata', {'type': 'raw_R'})
    x_math_data = best_fit_info.get('x_raw_data')
    t_data = best_fit_info.get('y_data_data')
    # Translate physical boundary coordinate into model domain coordinate
    r_trans_a = get_math_x_from_phys_r(r_phys_anchor, trans_meta)
    
    B1, B2 = best_fit_info.get('B1', 0.0), best_fit_info.get('B2', 1.0)
    stype = best_fit_info.get('scaling_type', 'none').lower()
    
    if stype == 'log': x_a = (np.log10(r_trans_a) - B1) / B2
    elif stype == 'linear': x_a = (r_trans_a - B1) / B2
    else: x_a = r_trans_a

    y_a = t_anchor
    original_deg = len(best_fit_info['params']) - 1
    candidate_result = None

    # Step-down degree reduction loop to find a stable constrained fit
    for current_deg in range(original_deg, 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_unc = np.polyfit(x_math_data, t_data, current_deg)

            # Define constraint equation: Y = Y_anchor + sum(Ai * (x^i - x_anchor^i))
            def constrained_model(x, *free_params):
                val = y_a
                for i, Ai in enumerate(free_params, start=1):
                    val += Ai * (np.power(x, i) - np.power(x_a, i))
                return val

            p0 = p_unc[::-1][1:] 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (OptimizeWarning, RuntimeWarning))
                popt, pcov = curve_fit(constrained_model, x_math_data, t_data, 
                                       p0=p0, sigma=best_fit_info.get('std_y_data'), 
                                       absolute_sigma=True, maxfev=5000)       
            
            # Resolve the dependent constant term (a0)
            a0 = y_a - sum(Ai * (x_a**i) for i, Ai in enumerate(popt, start=1))
            full_params = np.concatenate(([a0], popt))
            y_fit = np.polyval(full_params[::-1], x_math_data)
            
            if not np.all(np.isfinite(y_fit)) or np.any(np.abs(y_fit) > 1000):
                continue # Model exploded, try lower degree

            rmse_c = np.sqrt(np.mean((t_data - y_fit)**2))
            
            # Threshold acceptance: Discard if constraint warps the curve beyond 100mK)
            if rmse_c < 0.100:
                status = 'Numerical Optimal' if current_deg == original_deg else f'Deg Reduced ({current_deg})'
                if current_deg < original_deg:
                    logging.info(f"Knot {t_anchor:.2f}K: Reduced degree from {original_deg} to {current_deg} for stability.")
                
                candidate_result = _build_result_dict(best_fit_info, full_params, np.zeros_like(full_params), 
                                                     y_fit, t_data, y_a, r_phys_anchor, status)
                break
        except Exception:
            continue

    # Fallback 1: Rigid Delta Shift (Y-intercept adjustment only)
    if candidate_result is None:
        logging.info(f"Knot {t_anchor:.2f}K: Numerical solver failed. Trying Delta Shift.")
        p_fb = best_fit_info['params'].copy()
        delta = y_a - np.polyval(p_fb[::-1], x_a)
        p_fb[0] += delta
        y_fit_fb = np.polyval(p_fb[::-1], x_math_data)
        candidate_result = _build_result_dict(best_fit_info, p_fb, np.zeros_like(p_fb), 
                                             y_fit_fb, t_data, y_a, r_phys_anchor, 'Analytic Fallback')

    # Fallback 2: Complete Reversion to Unconstrained Model
    max_res = np.max(np.abs(candidate_result['residuals']))
    
    if max_res > 0.100:
        logging.warning(f"KNOT FAILED at {t_anchor:.2f}K (Max Resid: {max_res*1000:.1f} mK). Reverting to original fit.")
        reverted = best_fit_info.copy()
        reverted['is_constrained'] = False
        reverted['fit_status'] = 'Unconstrained (Knot Failed)'
        reverted['y_fit'] = np.polyval(best_fit_info['params'][::-1], x_math_data)
        reverted['residuals'] = t_data - reverted['y_fit']
        return reverted

    return candidate_result


def _build_result_dict(base, params, errors, y_fit, t_data, t_a, r_a, status):
    """Internal helper to reconstruct statistical diagnostics after forcing mathematical constraints."""
    resids = t_data - y_fit
    n, k = len(t_data), len(params)
    rss = np.sum(resids**2)
    
    dw = durbin_watson(resids)
    bp_p = np.nan
    try:
        X_mat = np.vander(base.get('x_raw_data'), k, increasing=True)
        if n > k:
            bp_p = het_breuschpagan(resids, X_mat)[1]
    except: pass

    updated = base.copy()
    updated.update({
        'params': params, 'param_errors': errors, 'y_fit': y_fit, 'residuals': resids,
        'reduced_chi_sq': (np.sum((resids / base.get('std_y_data'))**2)) / (n - k) if n > k else 0,
        'durbin_watson': dw, 'breusch_pagan_p': bp_p,
        'aic': 2*k + n*np.log(rss/n) if rss > 0 else np.inf,
        'is_constrained': True, 'fit_status': status, 'anchor_point': (t_a, r_a)
    })
    return updated

def find_optimal_knot_with_fallback(res1, res2, t_split, current_data):
    """
    Knot Optimizer: Searches for a micro-adjusted physical resistance coordinate (opt_r)
    near the boundary that minimizes the combined residual error of both segments, 
    smoothing the transition.
    """
    idx_knot = (np.abs(current_data['y'] - t_split)).argmin()
    r_phys_measured = current_data['x_untransformed'][idx_knot]
    
    # Estimate local sensor sensitivity (dT/dR) to convert temperature errors to resistance search bounds
    idx_p = max(0, idx_knot - 1)
    idx_n = min(len(current_data['y']) - 1, idx_knot + 1)
    dt = current_data['y'][idx_n] - current_data['y'][idx_p]
    dr = current_data['x_untransformed'][idx_n] - current_data['x_untransformed'][idx_p]
    slope = dt / dr if abs(dr) > 1e-10 else 1e-6

    def predict_t(res, r_target):
        # Interpolate the physical resistance back into the model's scaled numerical domain
        f_interp = interp1d(res['x_untransformed_data'], res['x_raw_data'], fill_value="extrapolate")
        x_target = f_interp(r_target)
        return np.polyval(res['params'][::-1], x_target)

    # Measure the unconstrained discontinuity gap
    gap_mk = abs(predict_t(res1, r_phys_measured) - predict_t(res2, r_phys_measured)) * 1000.0
    logging.info(f"Initial Knot Gap at {t_split}K: {gap_mk:.3f} mK")

    # Fast-path: Skip optimization if segments already align perfectly
    if gap_mk < 0.05:
        f1 = run_constrained_fit(res1, t_split, r_phys_measured, current_data)
        f2 = run_constrained_fit(res2, t_split, r_phys_measured, current_data)
        return f1, f2, r_phys_measured

    # Define search radius in Ohms equivalent to a 2mK - 5mK error margin
    search_mk = 5.0 if gap_mk > 5.0 else 2.0
    r_range = (search_mk / 1000.0) / abs(slope)

    def eval_node(r_test):
        f1_test = run_constrained_fit(res1, t_split, r_test, current_data)
        f2_test = run_constrained_fit(res2, t_split, r_test, current_data)
        rss_sum = np.sum(f1_test['residuals']**2) + np.sum(f2_test['residuals']**2)
        return rss_sum, f1_test, f2_test

    # Grid search for the optimal resistance anchor
    best_rss, opt_r, b_f1, b_f2 = float('inf'), r_phys_measured, None, None
    for test_r in np.linspace(r_phys_measured - r_range, r_phys_measured + r_range, 21):
        current_rss, f1_res, f2_res = eval_node(test_r)
        if current_rss < best_rss:
            best_rss, opt_r, b_f1, b_f2 = current_rss, test_r, f1_res, f2_res

    logging.info(f"Knot Optimized: Shift = {(opt_r - r_phys_measured):.6f} Ohm")
    return b_f1, b_f2, opt_r

# =============================================================================
# --- [SECTION 3: DATA EXPORT] ---
# =============================================================================

def save_stitched_dataset_to_csv(piecewise_results: list, config: dict):
    """
    Compiles the constrained mathematical vectors from all distinct segments 
    into a single, continuous DataFrame and exports it to a system CSV report.
    """
    base_dir = config['main_output_folder']
    model_name = config.get('analysis_params', {}).get('model_type', 'Piecewise_Model')
    
    all_data = []
    
    for i, res in enumerate(piecewise_results):
        t_meas = res.get('y_data_data')
        r_meas = res.get('x_untransformed_data')
        t_fit = res.get('y_fit')
        res_k = res.get('residuals')
        
        if t_meas is None or r_meas is None or t_fit is None:
            continue
            
        res_mk = res_k * 1000.0
        
        for t, r, tf, rk, rmk in zip(t_meas, r_meas, t_fit, res_k, res_mk):
            all_data.append({
                'Segment': i + 1,
                'T_measured_K': t,
                'R_measured_Ohm': r,
                'T_fitted_K': tf,
                'Residual_K': rk,
                'Residual_mK': rmk
            })
            
    if all_data:
        df = pd.DataFrame(all_data)
        df.sort_values(by='T_measured_K', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        out_path = os.path.join(base_dir, f"{model_name}_stitched_full_data.csv")
        df.to_csv(out_path, sep=';', index=False, float_format="%.8f")
        logging.info(f"Stitched dataset exported to: {out_path}")