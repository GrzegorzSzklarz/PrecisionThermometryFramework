# -*- coding: utf-8 -*-
"""
subset_generator.py - Data Partitioning & Variability Testing Module

This module provides a robust suite of functions for generating specialized 
subsets of a given dataset. These subsets are critical for:
- Sensitivity Analysis: Testing how model parameters fluctuate with reduced data.
- Cross-Validation: Evaluating model robustness across different thermal regimes.
- Outlier Stability Testing: Analyzing the impact of sequential outlier removal 
  on the Goodness-of-Fit (GoF) metrics.

Design Principles:
------------------
1. Index Tracking: Every generation function returns the array of indices used. 
   This ensures that associated metadata (like uncertainties or untransformed 
   resistances) can be synchronized perfectly with the generated subset.
2. Metrological Relevance: Supports systematic sampling (N-th point) to simulate 
   reduced calibration points, and temperature-band sampling
"""

import os
import logging
import numpy as np
import fitting_analysis_scripts.data_saver as data_saver
import fitting_analysis_scripts.dataset_combiner as dataset_combiner
import fitting_analysis_scripts.plotter as plotter
import copy

# --- Default Configuration Constants ---
DEFAULT_RANDOM_SUBSET_COUNT = 3
DEFAULT_RANDOM_SUBSET_SIZE = 30
DEFAULT_NTH_POINT_STEP = 3
DEFAULT_TEMP_THRESHOLD_CONFIGS = [
    {'temp_max': 15.0, 'step': 1},
    {'temp_min': 15.0, 'step': 2}
]
DEFAULT_NUM_OUTLIERS_TO_REMOVE = 5
DEFAULT_FIXED_DEGREE_FOR_OUTLIER_VARIABILITY = 13
DEFAULT_MAX_OUTLIERS_FOR_VARIABILITY_TEST = 5

# ==========================================
# --- 1. CORE GENERATION FUNCTIONS ---
# ==========================================

def _generate_random_subset_indices(num_points_full: int, size: int) -> np.ndarray:
    """Internal helper to generate a random sample of unique indices."""
    if size > num_points_full:
        print(f"Warning: Subset size ({size}) is larger than the dataset ({num_points_full}). Using all points.")
        size = num_points_full
    return np.random.choice(num_points_full, size=size, replace=False)

def generate_subset_by_criteria(y_data_full: np.ndarray, std_y_full: np.ndarray,
                                x_raw_full: np.ndarray, std_x_full: np.ndarray,
                                method: str, **kwargs) -> tuple:
    """
    Generates a single data subset based on predefined stochastic criteria.
    Currently supports Monte-Carlo style 'random' sampling.
    """
    num_points_full = len(y_data_full)
    if method == "random":
        size = kwargs.get('size', 10)
        chosen_indices = _generate_random_subset_indices(num_points_full, size)
    else:
        raise ValueError(f"Unknown subset_type: {method}")

    print(f"Generated a subset of type '{method}' with {len(chosen_indices)} points.")
    return (y_data_full[chosen_indices], std_y_full[chosen_indices],
            x_raw_full[chosen_indices], std_x_full[chosen_indices], chosen_indices)

def generate_nth_point_subsets(y_data_full: np.ndarray, std_y_full: np.ndarray,
                               x_raw_full: np.ndarray, std_x_full: np.ndarray,
                               step: int) -> list:
    """
    Generates multiple subsets using systematic sampling (every N-th point).
    Useful for testing calibration models with reduced point density.
    
    Returns:
        list of tuples: Each tuple contains (y, std_y, x, std_x, indices) for a phase.
    """
    if step <= 0: raise ValueError("Step must be a positive integer.")
    
    subsets = []
    num_points_full = len(y_data_full)
    for start_index in range(step):
        indices = np.arange(start_index, num_points_full, step)
        if indices.size > 0:
            subsets.append((y_data_full[indices], std_y_full[indices],
                            x_raw_full[indices], std_x_full[indices], indices))
            print(f"  Nth-point subset (start={start_index}, step={step}) generated with {len(indices)} points.")
    return subsets

def generate_temp_threshold_subsets(y_data_full: np.ndarray, std_y_full: np.ndarray,
                                    x_raw_full: np.ndarray, std_x_full: np.ndarray,
                                    configs: list) -> tuple:
    """
    Generates a unified subset by extracting data from specific temperature bands.
    Allows for differential sampling densities (steps) in different thermal regimes.
    """
    combined_indices = np.array([], dtype=int)
    
    for config in configs:
        mask = np.ones(len(y_data_full), dtype=bool)
        if 'temp_min' in config: mask &= (y_data_full >= config['temp_min'])
        if 'temp_max' in config: mask &= (y_data_full <= config['temp_max'])
        
        current_indices = np.where(mask)[0]
        step = config.get('step', 1)
        if step > 1:
            current_indices = current_indices[::step]
        combined_indices = np.union1d(combined_indices, current_indices)
    
    print(f"  Temp-threshold subset generated with {len(combined_indices)} points.")
    return (y_data_full[combined_indices], std_y_full[combined_indices],
            x_raw_full[combined_indices], std_x_full[combined_indices], combined_indices)

def generate_outlier_removed_subset(y_data_full: np.ndarray, std_y_full: np.ndarray,
                                    x_raw_full: np.ndarray, std_x_full: np.ndarray,
                                    residuals: np.ndarray, num_to_remove: int) -> tuple:
    """
    Creates a cleaner dataset by hard-rejecting the N points with the 
    largest absolute residuals from a previous model fit.
    """
    
    if num_to_remove < 0: raise ValueError("Number of outliers to remove cannot be negative.")
    if num_to_remove >= len(y_data_full):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    outlier_indices = np.argsort(np.abs(residuals))[-num_to_remove:]
    keep_indices = np.setdiff1d(np.arange(len(y_data_full)), outlier_indices, assume_unique=True)
    
    print(f"  Outlier-removal subset generated with {len(keep_indices)} points.")
    return (y_data_full[keep_indices], std_y_full[keep_indices],
            x_raw_full[keep_indices], std_x_full[keep_indices], keep_indices)

def generate_outlier_variability_subsets(y_data_full: np.ndarray, std_y_full: np.ndarray,
                                         x_raw_full: np.ndarray, std_x_full: np.ndarray,
                                         residuals: np.ndarray, max_outliers: int) -> list:
    """
    Generates a progressive sequence of subsets, iteratively removing the 
    largest residual point one by one. 
    Crucial for identifying the breakdown point where model metrics stabilize.
    """
    
    if max_outliers < 1: raise ValueError("max_outliers must be at least 1.")
    
    subsets = []
    sorted_outlier_indices = np.argsort(np.abs(residuals))[::-1]
    
    for i in range(1, max_outliers + 1):
        if i >= len(y_data_full): break
        
        removed_indices_so_far = sorted_outlier_indices[:i]
        keep_indices = np.setdiff1d(np.arange(len(y_data_full)), removed_indices_so_far, assume_unique=True)
        
        print(f"  Variability subset with {i} outliers removed generated ({len(keep_indices)} points).")
        subsets.append((y_data_full[keep_indices], std_y_full[keep_indices],
                        x_raw_full[keep_indices], std_x_full[keep_indices], 
                        removed_indices_so_far))
    return subsets

def generate_subset_by_removing_indices(y_data: np.ndarray, std_y: np.ndarray,
                                        x_raw: np.ndarray, std_x: np.ndarray,
                                        indices_to_remove: list) -> tuple:
    """Utility function to slice arrays by a specific list of indices."""
    y_data_new = np.delete(y_data, indices_to_remove)
    std_y_new = np.delete(std_y, indices_to_remove)
    x_raw_new = np.delete(x_raw, indices_to_remove)
    std_x_new = np.delete(std_x, indices_to_remove)
    return y_data_new, std_y_new, x_raw_new, std_x_new

# ==========================================
# --- 2. INTERACTIVE CLI WORKFLOW ---
# ==========================================

def run_subset_analysis_loop(current_data: dict, analysis_results: dict, config: dict):
    """
    Orchestrates an interactive CLI session for subset generation and immediate re-analysis.
    
    Features:
    - Supports both monolithic (Global) and segmented (Piecewise) models.
    - Dynamically reconstructs piecewise topologies for generated subsets.
    - Routes execution to the appropriate mathematical solver based on config.
    """
    
    # --- 1. DYNAMIC FOLDER NAMING ---
    current_n = current_data['num_points']
    folder_name_main = f"subsets_{current_n}pts"
    subsets_base_dir = os.path.join(config.get('main_output_folder', ''), folder_name_main)
    os.makedirs(subsets_base_dir, exist_ok=True)
    
    # Extract operational context
    run_analysis_func = config.get('run_analysis_func')
    analysis_params = config.get('analysis_params', {})
    p_mode = config.get('piecewise_mode', 'none')
    
    x_untransformed_full = current_data.get('x_untransformed', current_data['x'])

    # --- 2. RESIDUAL PREPARATION (FOR OUTLIER METHODS) ---
    best_fit_info = analysis_results.get('best_fit')
    all_residuals = []
    
    if best_fit_info is not None:
        if isinstance(best_fit_info, list):
            # Smooth piecewise discontinuities by averaging boundary residuals
            for i, seg in enumerate(best_fit_info):
                seg_res = list(seg.get('residuals', []))
                if i > 0 and len(seg_res) > 0:
                    all_residuals[-1] = (all_residuals[-1] + seg_res[0]) / 2.0
                    all_residuals.extend(seg_res[1:])
                else:
                    all_residuals.extend(seg_res)
            all_residuals = np.array(all_residuals)
        else:
            all_residuals = best_fit_info.get('residuals', [])

    # --- 3. INTERNAL ROUTING ENGINE ---
    def _execute_subset_analysis(label, y_s, sy_s, x_s, sx_s, x_untransformed_s, output_dir, override_params=None):
        """Routes execution to Global or Piecewise engines. Reconstructs segment topology if needed."""
        if p_mode in ['divided', 'combined']:
            # Package the flat subset into a dictionary
            subset_data = {
                'y': y_s, 'std_y': sy_s, 'x': x_s, 'std_x': sx_s,
                'x_untransformed': x_untransformed_s,
                'num_points': len(y_s), 'label': label
            }
            
            best_fit_models = analysis_results.get('best_fit')
            
            if isinstance(best_fit_models, list) and len(best_fit_models) > 1:
                subset_data['segments'] = []
                subset_data['piecewise_splits_T'] = current_data.get('piecewise_splits_T', [])
                subset_data['piecewise_num_funcs'] = current_data.get('piecewise_num_funcs', len(best_fit_models))
                
                print(f"--- Debug: Reconstructing {len(best_fit_models)} segments using splits: {subset_data['piecewise_splits_T']} ---")
                
                # Topological Reconstruction: Assign subset points to their original physical segments
                for i, orig_seg in enumerate(best_fit_models):
                    seg_y_data = orig_seg.get('y_data_data', orig_seg.get('y_raw', np.array([])))
                    if len(seg_y_data) == 0: continue
                        
                    t_min, t_max = np.min(seg_y_data), np.max(seg_y_data)
                    mask = (subset_data['y'] >= t_min - 1e-6) & (subset_data['y'] <= t_max + 1e-6)
                    
                    if np.any(mask):
                        seg_dict = {
                            'y': subset_data['y'][mask],
                            'x': subset_data['x'][mask],
                            'std_y': subset_data['std_y'][mask],
                            'std_x': subset_data['std_x'][mask],
                            'x_untransformed': subset_data['x_untransformed'][mask],
                            'label': f"{label}_Seg_{i+1}",
                            'num_points': np.sum(mask)
                        }
                        subset_data['segments'].append(seg_dict)

            local_config = copy.deepcopy(config)
            local_config['main_output_folder'] = output_dir
            
            print(f"[*] Running Piecewise Analysis for subset: {label}")
            pw_best, pw_all = dataset_combiner.run_intelligent_piecewise_analysis(subset_data, local_config)
            
            if pw_best:
                plotter.plot_piecewise_summary(pw_best, subset_data, pw_all, local_config)
                data_saver.save_piecewise_results(pw_best, subset_data, local_config)
        else:
            # Standard Monolithic Fitting
            current_params = override_params if override_params else analysis_params
            run_analysis_func(
                data_label=label, y_data_set=y_s, std_y_set=sy_s, x_raw_set=x_s, std_x_set=sx_s, 
                x_untransformed_set=x_untransformed_s, output_dir=output_dir, **current_params
            )
   
    # --- 4. CLI INTERFACE ---
    while True:
        print("\n--- Subset Analysis Menu ---")
        logging.info(f"Starting subset session on '{current_data['label']}' ({current_n} points).")
        
        print("1. Random Subsets")
        print("2. Nth Point Subsets (Systematic sampling)")
        print("3. Temperature Threshold Subsets (Segmented sampling)")
        print("4. Outlier Removal Subset (Based on current residuals)")
        print("5. Outlier Variability Test (Sequential removal)")
        print("0. Back to Further Analysis Menu")
        
        choice = input("Select subset method (or 0 to exit): ").strip()

        # --- OPTION 1: RANDOM SUBSETS ---
        if choice == '1':
            try:
                num_subsets = int(input(f"Number of random subsets (default {DEFAULT_RANDOM_SUBSET_COUNT}): ").strip() or DEFAULT_RANDOM_SUBSET_COUNT)
                subset_size = int(input(f"Subset size (points, default {DEFAULT_RANDOM_SUBSET_SIZE}): ").strip() or DEFAULT_RANDOM_SUBSET_SIZE)
                
                for i in range(num_subsets):
                    y_s, sy_s, x_s, sx_s, indices = generate_subset_by_criteria(
                        current_data['y'], current_data['std_y'], current_data['x'], current_data['std_x'], "random", size=subset_size)
                    
                    x_untransformed_s = x_untransformed_full[indices]
                    folder_name = f"subset-{len(y_s)}pts_Random_Run{i + 1}"
                    label = f"Random_Sub_{i + 1}_{len(y_s)}pts"
                    output_run_dir = os.path.join(subsets_base_dir, folder_name)
                    os.makedirs(output_run_dir, exist_ok=True)
                    
                    _execute_subset_analysis(label, y_s, sy_s, x_s, sx_s, x_untransformed_s, output_run_dir)
            except ValueError: 
                print("[Error] Invalid numeric input for Random Subsets.")

        # --- OPTION 2: N-TH POINT SUBSETS ---
        elif choice == '2':
            try:
                nth_step = int(input(f"Enter step N (default {DEFAULT_NTH_POINT_STEP}): ").strip() or DEFAULT_NTH_POINT_STEP)
                nth_subsets_list = generate_nth_point_subsets(
                    current_data['y'], current_data['std_y'], current_data['x'], current_data['std_x'], nth_step)
                
                for i, (y_s, sy_s, x_s, sx_s, indices) in enumerate(nth_subsets_list):
                    x_untransformed_s = x_untransformed_full[indices]
                    folder_name = f"subset-{len(y_s)}pts_Nth_Step{nth_step}_Start_{i}"
                    label = f"Nth_Point_Step{nth_step}_S{i}"
                    output_run_dir = os.path.join(subsets_base_dir, folder_name)
                    os.makedirs(output_run_dir, exist_ok=True)
                    
                    _execute_subset_analysis(label, y_s, sy_s, x_s, sx_s, x_untransformed_s, output_run_dir)
            except ValueError: 
                print("[Error] Invalid step value.")

        # --- OPTION 3: TEMPERATURE THRESHOLDS ---
        elif choice == '3':
            user_configs = []
            if (input(f"Use default threshold config? {DEFAULT_TEMP_THRESHOLD_CONFIGS} (y/n): ").strip().lower() == 'n'):
                while True:
                    print(f"\n--- Defining Temperature Segment #{len(user_configs) + 1} ---")
                    config_seg = {}
                    try:
                        min_str = input("MIN temp (blank for none): ").strip()
                        if min_str: config_seg['temp_min'] = float(min_str.replace(',', '.'))
                        max_str = input("MAX temp (blank for none): ").strip()
                        if max_str: config_seg['temp_max'] = float(max_str.replace(',', '.'))
                        step_str = input("Step for this segment (default 1): ").strip()
                        config_seg['step'] = int(step_str) if step_str else 1
                        
                        user_configs.append(config_seg)
                        if input("Add another segment? (y/n): ").strip().lower() != 'y': break
                    except ValueError as e:
                        print(f"[Error] Invalid threshold input: {e}")
            
            temp_configs_to_use = user_configs if user_configs else DEFAULT_TEMP_THRESHOLD_CONFIGS
            y_s, sy_s, x_s, sx_s, indices = generate_temp_threshold_subsets(
                current_data['y'], current_data['std_y'], current_data['x'], current_data['std_x'], temp_configs_to_use)
            
            if len(y_s) > 0:
                x_untransformed_s = x_untransformed_full[indices]
                folder_name = f"subset-{len(y_s)}pts_Temp_Threshold"
                label = f"Temp_Threshold_{len(y_s)}pts"
                output_run_dir = os.path.join(subsets_base_dir, folder_name)
                os.makedirs(output_run_dir, exist_ok=True)
                
                _execute_subset_analysis(label, y_s, sy_s, x_s, sx_s, x_untransformed_s, output_run_dir)

        # --- OPTION 4: TOP OUTLIERS REMOVAL ---
        elif choice == '4':
            if len(all_residuals) == 0:
                print("[Warning] No residuals found. Run a standard analysis first.")
                continue
            try:
                num_outliers = int(input(f"Outliers to remove (default {DEFAULT_NUM_OUTLIERS_TO_REMOVE}): ").strip() or DEFAULT_NUM_OUTLIERS_TO_REMOVE)
                y_s, sy_s, x_s, sx_s, indices = generate_outlier_removed_subset(
                    current_data['y'], current_data['std_y'], current_data['x'], current_data['std_x'], 
                    all_residuals, num_outliers)
                
                x_untransformed_s = x_untransformed_full[indices]
                folder_name = f"subset-{len(y_s)}pts_Outliers_Removed_{num_outliers}"
                label = f"Out_Removed_{num_outliers}pts"
                output_run_dir = os.path.join(subsets_base_dir, folder_name)
                os.makedirs(output_run_dir, exist_ok=True)
                
                _execute_subset_analysis(label, y_s, sy_s, x_s, sx_s, x_untransformed_s, output_run_dir)
            except ValueError: 
                print("[Error] Invalid number of outliers.")

        # --- OPTION 5: SEQUENTIAL OUTLIER VARIABILITY TEST ---
        elif choice == '5':
            if len(all_residuals) == 0:
                print("[Warning] No residuals found for variability test.")
                continue
            try:
                is_rational = "rational" in str(run_analysis_func).lower()
                v_params = {}
                
                if is_rational and p_mode not in ['divided', 'combined']:
                    v_params['fixed_n'] = int(input("Enter fixed numerator degree n: ").strip())
                    v_params['fixed_m'] = int(input("Enter fixed denominator degree m: ").strip())
                    label_suffix = f"n{v_params['fixed_n']}_m{v_params['fixed_m']}"
                elif p_mode not in ['divided', 'combined']:
                    fixed_degree = int(input(f"Fixed degree (default {DEFAULT_FIXED_DEGREE_FOR_OUTLIER_VARIABILITY}): ").strip() or DEFAULT_FIXED_DEGREE_FOR_OUTLIER_VARIABILITY)
                    v_params['fixed_degree'] = fixed_degree
                    label_suffix = f"Deg{fixed_degree}"
                else:
                    label_suffix = "Piecewise"
                
                max_outliers = int(input("Max outliers to test (default 5): ").strip() or 5)
                
                variability_subsets = generate_outlier_variability_subsets(
                    current_data['y'], current_data['std_y'], current_data['x'], current_data['std_x'], 
                    all_residuals, max_outliers)
                
                for count, (y_s, sy_s, x_s, sx_s, removed_indices) in enumerate(variability_subsets, start=1):
                    x_untransformed_s = np.delete(x_untransformed_full, removed_indices)
                    folder_name = f"subset-{len(y_s)}pts_Variability_Step_{count}"
                    label = f"Variability_R{count}_{label_suffix}"
                    output_run_dir = os.path.join(subsets_base_dir, "Variability_Tests", folder_name)
                    os.makedirs(output_run_dir, exist_ok=True)
                    
                    current_v_args = {**analysis_params, **v_params}
                    
                    if is_rational and p_mode not in ['divided', 'combined']:
                        from fitting_analysis_scripts.rational_function_handler import _run_fixed_rational_fit_for_variability
                        _run_fixed_rational_fit_for_variability(
                            data_label=label, y_data_set=y_s, std_y_set=sy_s, x_raw_set=x_s, std_x_set=sx_s, 
                            x_untransformed_set=x_untransformed_s, output_dir=output_run_dir, **current_v_args
                        )
                    else:
                        _execute_subset_analysis(label, y_s, sy_s, x_s, sx_s, x_untransformed_s, output_run_dir, override_params=current_v_args)
            
            except ValueError: 
                print("[Error] Invalid numeric input.")
        
        elif choice == '0':
            break
        else:
            print("[!] Invalid option. Please select 0-5.")