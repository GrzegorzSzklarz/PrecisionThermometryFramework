# -*- coding: utf-8 -*-

"""
Calibration Framework - Main Execution Controller

This module serves as the central orchestrator for the precision thermometer 
calibration suite. It manages the lifecycle of numerical analysis, from initial 
data acquisition to iterative outlier removal and final reporting.

Core Functionalities:
---------------------
1. Data Acquisition & Transformation Persistence: 
   Standardizes CSV input and implements a state-lock mechanism for X-axis 
   transformations (e.g., W, ln(W)). Ensures mathematical consistency after 
   outlier removal by auto-reapplying chosen transformations.

2. Statistical Rigor & Model Selection:
   - Information Criteria: Uses AIC (Akaike) and BIC (Bayesian) to penalize 
     overfitting, following the N/3 rule (max parameters < 1/3 of data points).
   - Residual Diagnostics: Includes the Durbin-Watson test for autocorrelation 
     and the Breusch-Pagan test for heteroscedasticity to validate fit assumptions.
   - Goodness-of-Fit: Calculates Reduced Chi-Squared to assess if residuals 
     are consistent with the provided measurement uncertainties (std_y).

3. Dataset Manipulation & Segmented Analysis:
   - Piecewise Analysis: Allows dividing a dataset into temperature segments 
     for localized high-precision fitting (Intelligent Piecewise Workflow).
   - Dataset Combining: Enables merging multiple calibration runs into a single 
     global dataset for stability analysis or wide-range modeling.
   - Comparative Analysis: Provides tools to compare residuals between different 
     datasets to evaluate sensor drift or calibration reproducibility.

4. Iterative Outlier Analysis: 
   Integrates an interactive loop for data cleaning using Z-score, IQR, 
   and Studentized Residuals, enabling real-time model updates.

Numerical Workflow:
-------------------
Step 1: Environment resolution and file selection.
Step 2: Model selection and dynamic complexity limit calculation.
Step 3: Workflow routing (Rational, ITS-90, or Standard).
Step 4: Persistence layer setup and numerical engine execution.
Step 5: Statistical validation (Diagnostic tests and Information Criteria).
Step 6: Interactive dataset manipulation (Cleaning, Splitting, or Combining).

Author: g.szklarz
Project: Direk-T Calibration Framework
Date: 2026-03-08
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fitting_analysis_scripts'))

import fitting_analysis_scripts.data_loader as data_loader
import fitting_analysis_scripts.data_saver as data_saver
import fitting_analysis_scripts.plotter as plotter
import fitting_analysis_scripts.analyzer as analyzer
import fitting_analysis_scripts.function_defs as function_defs
import fitting_analysis_scripts.logger_setup as logger_setup
import interactive_handlers as handlers
import fitting_analysis_scripts.rational_function_handler as rational_handler

# --- Global Configuration Section ---
DEFAULT_FITTING_FUNCTION_NAME = "Scaled Polynomial + Sine"

# --- Helper Function for Running Analysis ---
def run_single_analysis(data_label: str, y_data_set: np.ndarray, std_y_set: np.ndarray,
                        x_raw_set: np.ndarray, std_x_set: np.ndarray,
                        output_dir: str, **kwargs) -> dict:
    """
    A wrapper for the core analysis function from the analyzer module.

    This simplifies calls to the main analysis engine by pre-configuring
    the plotter and saver functions, making the main workflow cleaner.

    Args:
        data_label (str): A descriptive label for the current analysis run.
        y_data_set, std_y_set, x_raw_set, std_x_set: The data arrays for analysis.
        output_dir (str): The explicit directory path for saving results.
        **kwargs: Other required parameters for the analysis (B1_val, etc.).

    Returns:
        dict: The dictionary of analysis results from the analyzer.
    """
    
    show_plots = kwargs.pop('show_plots', True)
    
    if show_plots:
        actual_plotter = plotter.plot_analysis_results
    else:
        actual_plotter = lambda *args, **kw: None
    
    return analyzer.perform_analysis_and_save_results(
        data_label=data_label, y_data_set=y_data_set, std_y_set=std_y_set,
        x_raw_set=x_raw_set, std_x_set=std_x_set,
        output_dir=output_dir,
        plotter_func=actual_plotter,
        saver_func_stats=data_saver.save_statistics,
        saver_func_params=data_saver.save_parameters,
        saver_func_best_fit=data_saver.save_best_fit_results,
        **kwargs 
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Step 0: Initial Environment Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, 'data')
    
    # Global default for the model selection interface
    DEFAULT_FITTING_FUNCTION_NAME = "Scaled Polynomial + Sine"

    # Principal application loop - allows batch processing of multiple thermometer files
    while True:
        # --- Step 1A: Data Acquisition & File Selection ---
        # Prompt the user for the data folder; defaults to the /data subfolder
        data_folder = handlers.get_data_folder_path(default_path=default_data_path)
        
        # Search for all CSV files in the selected directory
        csv_files = glob.glob(os.path.join(data_folder, '*.csv'))

        if not csv_files:
            print(f"Error: No CSV files found in: {data_folder}")
            if input("Try another directory? (y/n): ").strip().lower() != 'y':
                break
            else:
                continue

        # Interactive file selector for the user
        selected_input_file = handlers.select_file_from_list(csv_files)
        if selected_input_file is None:
            break

        # --- Step 1B: Data Loading & Pre-processing ---
        # Initialize DataFrame and capture the absolute path for session metadata
        df_original, original_file_abs_path = data_loader.load_data(selected_input_file)
        
        # Thermometer identifier derived from the filename
        base_file_name_no_ext = os.path.splitext(os.path.basename(original_file_abs_path))[0]
        
        # --- DYNAMIC CALCULATION OF MODEL LIMITS ---
        # Rule of thumb: maximum degree is defined as $Degree_{max} = \min(\lfloor N/3 \rfloor + 1, 30)$
        num_points = len(df_original)
        current_max_degree = min((num_points // 3) + 1, 30)
        
        # --- Step 2: Global Model Selection ---
        # Dispatcher for standard fits, ITS-90 calibrations, or specialized models
        selected_mode = handlers.select_fitting_function(DEFAULT_FITTING_FUNCTION_NAME)
        func_info = function_defs.get_fitting_function(selected_mode)
        
        # --- Step 3: Workflow Routing (Branching) ---
        # CASE A: SPECIAL WORKFLOWS (Rational Functions / Pade Approximants)
        # This branch handles non-linear rational models using a full (n,m) scan.
        # Optimized for datasets requiring asymptotic behavior modeling.
        if func_info and func_info.get('is_special_workflow'):
            rational_handler.handle_rational_function_analysis(
                df_original=df_original,
                base_file_name_no_ext=base_file_name_no_ext,
                data_folder=data_folder,
                max_polynomial_degree=current_max_degree 
            )
            
        # CASE B: ITS-90 PRT CALIBRATION WORKFLOW
        # Implements the official ITS-90 deviation equations for PRTs.
        # Includes fixed-point calibration and reference function calculations.
        elif selected_mode == "ITS-90_CALIBRATION":
            current_data = {
                'y': df_original['T'].values, 'std_y': df_original['Tstd'].values,
                'x': df_original['R'].values, 'std_x': df_original['Rstd'].values
            }
            
            # Directory structure: /results/{Sensor_ID}/ITS-90_CALIBRATION/
            mode_foldername = handlers.sanitize_foldername(selected_mode)
            rel_path = os.path.join(base_file_name_no_ext, mode_foldername)
            main_output_folder = data_saver.get_global_results_path(rel_path)
            
            # Initialize specialized logging for this calibration session
            log_file_path = os.path.join(main_output_folder, 'analysis_log.txt')
            logger_setup.setup_logger(log_file_path)
            logging.info(f"Logger initialized for ITS-90. Path: {log_file_path}")
                        
            handler_config = {
                'base_file_name': base_file_name_no_ext,
                'main_output_folder': main_output_folder,
                'data_folder': data_folder,
                'analysis_params': {'analyzer_module': analyzer}
            }
            
            print("\nStarting ITS-90 Calibration procedure...")
            handlers.handle_its90_calibration(current_data, handler_config)
            
        # CASE C: STANDARD FITTING & INTERACTIVE DATASET MANAGEMENT
        # The primary workflow for polynomial/exponential fits. Supports:
        # - Outlier removal cycles.
        # - Piecewise analysis (Splitting data into T-segments).
        # - Merging datasets from the interactive menu.
        # - Comparing residuals across different measurement sessions.
        else:
            current_df = df_original.copy()
            
            # --- PERSISTENCE LAYER ---
            # These variables lock the transformation state (e.g., $W_{TPW}$ or $\ln(W)$).
            # Once selected, the transformation is automatically reapplied to cleaned datasets.
            saved_transformation_label = ""
            saved_transform_details = {'type': 'not_set'}

            # Iterative Analysis Loop: Repeats fitting until data cleaning is complete
            while True:
                B1_val = current_df['R'].min()
                B2_val = current_df['R'].max()
                num_points = len(current_df)
                
                # Snapshot of the current state of data
                current_data = {
                    'y': current_df['T'].values, 
                    'std_y': current_df['Tstd'].values,
                    'x': current_df['R'].values, 
                    'std_x': current_df['Rstd'].values,
                    'label': f"Data_{num_points}pts", 
                    'num_points': num_points,
                    'x_untransformed': current_df['R'].values.copy()
                }

                func_metadata = function_defs.get_fitting_function(selected_mode)
                scaling_type = func_metadata.get('scaling_type', 'none')

    # --- TRANSFORMATION PERSISTENCE LOGIC ---
                # Applied only to models without built-in normalization (e.g., standard polynomials)
                if scaling_type == 'none':
                    
                    # INITIAL SETUP: Prompt user for transformation once per session
                    if saved_transform_details['type'] == 'not_set':
                        new_x, new_std_x, x_label, t_details = handlers.handle_x_transformation(current_data)
                        
                        if new_x is not None:
                            current_data['x'] = new_x
                            current_data['std_x'] = new_std_x
                            saved_transformation_label = x_label
                            saved_transform_details = t_details
                        else:
                            # Default to Raw Resistance; enforce label for consistent folder naming
                            saved_transformation_label = "raw_R"
                            saved_transform_details = {'type': 'raw_R'}
                    
                    # RE-APPLICATION: Auto-calculate transformed X after point removal
                    elif saved_transform_details['type'] != 'raw_R':
                        t_type = saved_transform_details['type']
                        raw_x = current_data['x_untransformed']
                        raw_std = current_data['std_x']
                        
                        if t_type == 'ln_R':
                            current_data['x'] = np.log(raw_x)
                            current_data['std_x'] = raw_std / raw_x
                        elif t_type in ['W_TPW', 'W_Ne', 'W_Ar']:
                            r_ref = saved_transform_details['r_ref']
                            current_data['x'] = raw_x / r_ref
                            current_data['std_x'] = raw_std / r_ref
                        elif t_type in ['ln_W', 'ln_W_Ne', 'ln_W_Ar']:
                            r_ref = saved_transform_details['r_ref']
                            W_val = raw_x / r_ref
                            current_data['x'] = np.log(W_val)
                            current_data['std_x'] = (raw_std / r_ref) / W_val
                            
                    # Append transformation suffix to the dataset label for traceability
                    current_data['label'] = f"{current_data['label']}_{saved_transformation_label}"
                
                # --- DIRECTORY PATHING & LOGGING ---
                mode_foldername = handlers.sanitize_foldername(selected_mode)
                
                # Structure: results/{ID}/{Model}/{Transformation_Label}/
                if saved_transformation_label:
                    rel_path = os.path.join(base_file_name_no_ext, mode_foldername, saved_transformation_label)
                else:
                    rel_path = os.path.join(base_file_name_no_ext, mode_foldername)
                
                main_output_folder = data_saver.get_global_results_path(rel_path)
                os.makedirs(main_output_folder, exist_ok=True)
                             
                logger_setup.setup_logger(os.path.join(main_output_folder, 'analysis_log.txt'))
                    
                # Prepare parameters for the numerical analyzer
                analysis_params = {
                    'B1_val': B1_val, 
                    'B2_val': B2_val, 
                    'max_degree': current_max_degree,
                    'file_base_name': base_file_name_no_ext, 
                    'fitting_function_name': selected_mode,
                    'transform_details': saved_transform_details
                }

                handler_config = {
                    'base_file_name': base_file_name_no_ext, 
                    'main_output_folder': main_output_folder,
                    'mode_foldername': mode_foldername,
                    'data_folder': data_folder,
                    'run_analysis_func': run_single_analysis,
                    'analysis_params': analysis_params,
                    'x_transformation_metadata': saved_transform_details
                }
                
                # --- Primary Numerical Analysis ---
                print(f"--- Starting Full Data Analysis ({num_points} points) ---")
                full_analysis_results = run_single_analysis(
                    data_label=current_data['label'],
                    y_data_set=current_data['y'], 
                    std_y_set=current_data['std_y'], 
                    x_raw_set=current_data['x'], 
                    std_x_set=current_data['std_x'], 
                    x_untransformed_set=current_data['x_untransformed'],
                    output_dir= main_output_folder, 
                    **analysis_params
                )
                
                # Extract the optimal model based on Akaike Information Criterion (AIC)
                analysis_results = {'full': full_analysis_results, 'best_fit': None}
                if full_analysis_results:
                    best_degree = min(full_analysis_results, key=lambda k: full_analysis_results[k]['aic'])
                    analysis_results['best_fit'] = full_analysis_results[best_degree]

                # Persist the 4-column comprehensive global report
                data_saver.save_global_report(analysis_results['best_fit'], current_data, handler_config)
                
                # --- Outlier Analysis & Piecewise Loop ---
                # User menu for data cleaning (Z-score, IQR, etc.) or segmentation
                cleaned_data_dict, results, should_exit, should_reanalyze = handlers.run_fit_analysis_loop(
                    current_data, analysis_results, handler_config
                )
                
                if should_exit: 
                    print("\n[!] Program terminated by user.")
                    import sys
                    sys.exit()
                
                if not should_reanalyze:
                    print("\n[!] Returning to Main Menu (File/Function selection)...")
                    break
                
                if should_reanalyze:
                    current_df = pd.DataFrame({
                        'R': cleaned_data_dict['x_untransformed'],
                        'T': cleaned_data_dict['y'],
                        'Rstd': cleaned_data_dict.get('std_x', 0), 
                        'Tstd': cleaned_data_dict.get('std_y', 0)
                    })
                    
                    handler_config['piecewise_mode'] = 'none'
            
                    continue
                else:
                    break
                
        print(f"\n--- Final report for {base_file_name_no_ext} generated ---")

    print("\n--- Program execution finished ---")