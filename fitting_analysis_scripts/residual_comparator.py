"""
residual_comparator.py

This module validates mathematical models by evaluating them against independent 
datasets (e.g., comparing a model trained on a cooling cycle against data from 
a heating cycle).

Capabilities:
-------------
1. Model Projection: Accurately projects both Global and Piecewise models onto 
   new datasets, respecting physical segment boundaries and topological knots.
2. Signal Smoothing: Applies Rolling Window (Moving Average) filtering to 
   isolate systematic drift or hysteresis from high-frequency measurement noise.
3. Diagnostic Output: Generates comparative visual panels and unified CSV logs.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import fitting_analysis_scripts.data_loader as data_loader
import fitting_analysis_scripts.function_defs as function_defs
import interactive_handlers as handlers

def _calculate_comparison_residuals(comparison_df, best_fit_info, config):
    """
    Calculates residuals for a new dataset using an existing fit model.

    This robust function supports both Global Fit models (dict) and Piecewise 
    Fit models (list of dicts). It precisely replicates the original data 
    transformations (e.g., physical ratios like W_Ne, or numerical scaling like 
    linear normalization to [-1, 1]) on a per-segment basis to ensure a valid 
    mathematical comparison.

    Args:
        comparison_df (pd.DataFrame): The DataFrame of the new dataset to test.
        best_fit_info (dict or list): The result object from the original best fit. 
                                      A dictionary for global fits, or a list of dictionaries for piecewise.
        config (dict): The main configuration dictionary containing transformation 'recipes'.

    Returns:
        pd.DataFrame or None: The comparison DataFrame with added 'x_transformed' 
                              and 'residuals' columns, or None if the process fails.
    """

    y_comp = comparison_df['T'].values
    r_vals_comp = comparison_df['R'].values
    
    # Initialize output arrays
    y_predicted = np.zeros_like(y_comp, dtype=float)
    x_transformed_full = np.zeros_like(r_vals_comp, dtype=float)
    processed_mask = np.zeros_like(y_comp, dtype=bool)

    # 1. Determine if the model is Piecewise (segmented) or Global
    is_piecewise = isinstance(best_fit_info, list)
    models = best_fit_info if is_piecewise else [best_fit_info]

    # Get the global "recipe" for transformation/scaling
    analysis_params = config.get('analysis_params', {})
    # Handle both Standard (transform_details) and Rational (norm_params) keys
    transform_details = analysis_params.get('transform_details', analysis_params.get('norm_params', {}))

    def _evaluate_points(r_vals, model_info):
        """
        Internal helper function to apply mathematical transformations and 
        predict Temperature values for a specific subset of data using a specific model segment.
        """
# 1. Retrieve the appropriate mathematical function
        func_name = model_info.get('fitting_function_name', analysis_params.get('fitting_function_name'))
        if not func_name:
            if 'n' in model_info: 
                func_name = 'Rational Function (Pade-like)'
            else:
                logging.error("Could not determine the fitting function for projection.")
                return None, None

        x_for_fit = r_vals.copy()
        
        # =====================================================================
        # PATHWAY A: Rational Function Logic (Padé approximants)
        # =====================================================================
        if 'Rational' in func_name:
            choice = transform_details.get('choice')
            if not choice:
                logging.error("Normalization details not found for rational function.")
                return None, None
            
            # FIX: Extract frozen mathematical boundaries (Rmin/Rmax) from the original training segment
            seg_norm = model_info.get('norm_params', {})
            rmin = seg_norm.get('Rmin', model_info.get('Rmin', transform_details.get('Rmin')))
            rmax = seg_norm.get('Rmax', model_info.get('Rmax', transform_details.get('Rmax')))
            r_ref = seg_norm.get('r_ref', model_info.get('r_ref', transform_details.get('r_ref', 1.0)))

            # FALLBACK: If boundaries are somehow missing, extract them from the raw training data.
            # We strictly avoid using np.min(r_vals) to prevent domain shifting.
            if rmin is None or rmax is None:
                orig_x = model_info.get('x_untransformed_data', r_vals)
                rmin, rmax = np.min(orig_x), np.max(orig_x)
                logging.warning(f"Rmin/Rmax metadata missing. Recovered from training data: [{rmin}, {rmax}]")

            # Apply the specific topological normalization chosen during training
            if choice == 1:
                # Linear Scaling to [0, 1]
                x_for_fit = (r_vals - rmin) / (rmax - rmin) if (rmax - rmin) != 0 else np.zeros_like(r_vals)
            elif choice == 5:
                # Logarithmic Scaling
                ln_r, ln_rmin, ln_rmax = np.log(r_vals), np.log(rmin), np.log(rmax)
                x_for_fit = (ln_r - ln_rmin) / (ln_rmax - ln_rmin) if (ln_rmax - ln_rmin) != 0 else np.zeros_like(r_vals)
            elif choice in [2, 3, 4]:
                # Simple Ratio (e.g., W = R / R_tpw)
                x_for_fit = r_vals / r_ref
            elif choice in [6, 7, 8]:
                # Natural Log of Ratio (e.g., ln(W))
                x_for_fit = np.log(r_vals / r_ref)

        # =====================================================================
        # PATHWAY B: Standard Functions (Polynomials, Exponentials)
        # =====================================================================
        else:
            x_comp_temp = r_vals.copy()
            
            # Step B1: Apply initial physical transformation (e.g., converting Ohms to W ratio)
            if transform_details:
                ttype = transform_details.get('type')
                r_ref = transform_details.get('r_ref', 1.0)
                if ttype in ['W', 'W_Ne', 'W_Ar']:
                    x_comp_temp = r_vals / r_ref
                elif ttype in ['ln_W', 'ln_W_Ne', 'ln_W_Ar']:
                    x_comp_temp = np.log(r_vals / r_ref)
                elif ttype == 'ln_R':
                    x_comp_temp = np.log(r_vals)
            
            # Step B2: Apply secondary numerical scaling (e.g., compressing domain to [-1, 1] for Chebyshev)
            func_info = function_defs.get_fitting_function(func_name)
            scaling_type = func_info.get('scaling_type', 'none') if func_info else 'none'
            x_for_fit = x_comp_temp
            
            if scaling_type != 'none':
                # FIX: Retrieve frozen domain bounds (B1, B2) established during model training
                B1 = model_info.get('B1_val', analysis_params.get('B1_val'))
                B2 = model_info.get('B2_val', analysis_params.get('B2_val'))
                
                # FALLBACK: Recover bounds from original training data if metadata is missing
                if B1 is None or B2 is None:
                    orig_x_trans = model_info.get('x_raw', x_comp_temp)
                    B1, B2 = np.min(orig_x_trans), np.max(orig_x_trans)
                    logging.warning(f"B1/B2 bounds missing. Recovered from training data: [{B1}, {B2}]")
                
                # Execute scaling logic
                if scaling_type == 'linear':
                    x_for_fit = (2 * x_comp_temp - B1 - B2) / (B2 - B1) if (B2 - B1) != 0 else np.zeros_like(x_comp_temp)
                elif scaling_type == 'log':
                    ln_x, ln_xmin, ln_xmax = np.log(x_comp_temp), np.log(B1), np.log(B2)
                    denominator = ln_xmax - ln_xmin
                    x_for_fit = (2 * ln_x - ln_xmax - ln_xmin) / denominator if denominator != 0 else np.zeros_like(x_comp_temp)

        # =====================================================================
        # FINAL STEP: Predict Y values (Temperature Output)
        # =====================================================================
        y_pred = None
        
        if 'Rational' in func_name:
            # Construct the specific Padé topology using trained degrees
            n, m, b0 = model_info['n'], model_info['m'], model_info.get('b0_is_zero', True)
            f = function_defs.create_rational_function(n, m, b0)
            y_pred = f(x_for_fit, *model_info['params'])
        else:
            # Dispatch to standard mathematical libraries
            func_info = function_defs.get_fitting_function(func_name)
            if func_info: 
                y_pred = func_info['function'](x_for_fit, *model_info['params'])
            else:
                logging.error(f"Function '{func_name}' not found in registry during projection.")
        
        return x_for_fit, y_pred

    # 2. Iterate through available models (1 for Global, multiple for Piecewise)
    for seg in models:
        if is_piecewise:
            # Determine the temperature bounds of the current segment
            t_min = seg.get('y_data_data', seg.get('y_raw', np.array([-np.inf]))).min()
            t_max = seg.get('y_data_data', seg.get('y_raw', np.array([np.inf]))).max()
            
            # Condition `~processed_mask` ensures overlapping boundaries don't process points twice
            mask = (y_comp >= t_min) & (y_comp <= t_max) & ~processed_mask
        else:
            # Global model processes all unprocessed points
            mask = ~processed_mask
            
        if not np.any(mask):
            continue 
            
        r_vals_seg = r_vals_comp[mask]
        
        # Evaluate points using the segment's specific parameters
        x_trans_seg, y_pred_seg = _evaluate_points(r_vals_seg, seg)
        
        if y_pred_seg is not None:
            x_transformed_full[mask] = x_trans_seg
            y_predicted[mask] = y_pred_seg
            processed_mask[mask] = True

    # Handle Extrapolation: Assign out-of-bounds points to the geometrically closest segment
    unprocessed_indices = np.where(~processed_mask)[0]
    if len(unprocessed_indices) > 0 and is_piecewise:
        for idx in unprocessed_indices:
            t_val = y_comp[idx]
            r_val = np.array([r_vals_comp[idx]]) # Pass as an array of 1 element
            
            # Find the closest segment based on temperature distance
            closest_seg = min(models, key=lambda s: min(
                abs(s.get('y_data_data', np.array([0])).min() - t_val), 
                abs(s.get('y_data_data', np.array([0])).max() - t_val)
            ))
            
            x_trans_seg, y_pred_seg = _evaluate_points(r_val, closest_seg)
            if y_pred_seg is not None:
                x_transformed_full[idx] = x_trans_seg[0]
                y_predicted[idx] = y_pred_seg[0]
                processed_mask[idx] = True

    # Final validation
    if not np.all(processed_mask):
        logging.error("Failed to predict Y values for some points in the comparison dataset.")
        return None

    # Populate the dataframe with results
    comparison_df['x_transformed'] = x_transformed_full
    comparison_df['y_pred'] = y_predicted
    comparison_df['residuals'] = y_comp - y_predicted
    
    return comparison_df

def _calculate_moving_average(data_series, window_size):
    """Applies a centered moving average filter to suppress high-frequency noise."""
    return data_series.rolling(window=window_size, center=True, min_periods=1).mean()

def _plot_and_save_comparison(original_data, comp_data, x_col_orig, x_col_comp, y_col_orig, y_col_comp, title, ylabel, filename, xlabel="Temperature, K", splits=None):
    """
    Generates a standardized metrological comparison plot between the training 
    dataset and the validation dataset.

    Args:
        original_data (dict): The result dictionary for the original fit.
        comp_data (pd.DataFrame): The DataFrame for the comparison dataset.
        y_col_orig (str): The name of the column to plot for the original data.
        y_col_comp (str): The name of the column to plot for the comparison data.
        title (str): The title for the plot.
        ylabel (str): The Y-axis label for the plot.
        filename (str): The full path to save the output PNG file.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(original_data[x_col_orig], original_data[y_col_orig] * 1000, 'o', ms=5, alpha=0.7, label='Original Data (rediuals)')
    ax.plot(comp_data[x_col_comp], comp_data[y_col_comp] * 1000, 'x', ms=5, alpha=0.7, label='Comparison Data (rediuals)')
    
    if splits:
        for i, split in enumerate(splits):
            label = 'Segment Boundaries' if i == 0 else None
            ax.axvline(x=split, color='gray', linestyle='--', alpha=0.5, label=label)
    
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    plt.show(block=False)

    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Comparison plot saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving comparison plot: {e}")

def _save_comparison_csv(original_data: dict, comp_data: pd.DataFrame, avg_win_size: int, output_dir: str, file_base_name: str):
    """
    Persists cross-validation results to comprehensive CSV files.
    Safely merges arrays of different lengths (Training vs Validation sets) 
    using pandas concatenation to prevent ValueError exceptions.

    Args:
        original_data (dict): Result dictionary for the original fit, including averaged residuals.
        comp_data (pd.DataFrame): DataFrame for the comparison data, including averaged residuals.
        avg_win_size (int): The window size used for averaging.
        output_dir (str): The directory path to save the CSV files.
        file_base_name (str): The base name for the output files.
    """
    try:
        # --- Save Raw Residuals ---
        df_orig = pd.DataFrame({
            'R_original': pd.Series(original_data['x_untransformed_data']),
            'Residual_original': pd.Series(original_data['residuals'])
        })
        
        df_comp = pd.DataFrame({
            'R_comparison': pd.Series(comp_data['R'].values),
            'Residual_comparison': pd.Series(comp_data['residuals'].values),
        })
        
        df_raw_combined = pd.concat([df_orig, df_comp], axis=1)
        raw_path = os.path.join(output_dir, f"{file_base_name}_residuals_comparison.csv")
        df_raw_combined.to_csv(raw_path, sep=',', index=False, float_format='%.8f')
        logging.info(f"Raw residual comparison data saved to: {raw_path}")
        
        # --- Save Averaged Residuals ---
        df_orig_avg = pd.DataFrame({
            'R_original_avg': original_data['x_untransformed_data'],
            'Residual_original_avg': pd.Series(original_data['residuals_avg']).reset_index(drop=True)
        })
        
        df_comp_avg = pd.DataFrame({
            'R_comparison_avg': comp_data['R'].values,
            'Residual_comparison_avg': comp_data['residuals_avg'].reset_index(drop=True)
        })
        
        df_avg_combined = pd.concat([df_orig_avg, df_comp_avg], axis=1)
        avg_path = os.path.join(output_dir, f"{file_base_name}_residuals_comparison_avg_{avg_win_size}pts.csv")
        df_avg_combined.to_csv(avg_path, sep=',', index=False, float_format='%.8f')
        logging.info(f"Averaged residual comparison data saved to: {avg_path}")
    except Exception as e:
        logging.error(f"Failed to save comparison CSV files: {e}", exc_info=True)

def run_comparison(best_fit_info: dict, config: dict):
    """
    Orchestrates the interactive cross-validation workflow.
    Allows the user to select a validation dataset and define filtering parameters.

    Args:
        best_fit_info (dict): The result dictionary from the original best fit,
                              containing the model parameters and original data.
        config (dict): The main configuration dictionary, providing access to paths
                       and analysis parameters.
    """
    print("\n--- Compare Residuals with another Dataset ---")
    
    # Step 1: Select the comparison file
    data_folder = config['data_folder']
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    if not csv_files:
        print("No CSV files found in the data directory to compare with.")
        return
        
    comp_file_path = handlers.select_file_from_list(csv_files)
    if not comp_file_path:
        return # User chose to exit

    # Step 2: Get averaging window size from the user
    try:
        n_str = input("Enter the number of points (N) for the moving average: ").strip()
        avg_win_size = int(n_str)
        if avg_win_size < 2:
            print("Warning: Window size must be at least 2. Using N=2.")
            avg_win_size = 2
    except (ValueError, TypeError):
        print("Invalid number. Aborting comparison.")
        return

    # Step 3: Load and process the comparison dataset
    comparison_df, _ = data_loader.load_data(comp_file_path)
    if comparison_df is None or comparison_df.empty:
        print("Failed to load or process comparison data.")
        return
        
    comparison_df = _calculate_comparison_residuals(comparison_df, best_fit_info, config)
    if comparison_df is None:
        return

    # Step 4: Perform moving average calculations on both datasets
    original_results = {}
    is_piecewise = isinstance(best_fit_info, list)
    
    if is_piecewise:
        # Vectorized segment stitching (replaces slow Python loop)
        original_results['residuals'] = np.concatenate([s['residuals'] 
            if i==0 
            else s['residuals'][1:] for i, s in enumerate(best_fit_info)])
        
        original_results['y_data_data'] = np.concatenate([s.get('y_data_data', s.get('y_raw', [])) 
            if i==0 
            else s.get('y_data_data', s.get('y_raw', []))[1:] for i, s in enumerate(best_fit_info)])
        
        original_results['x_untransformed_data'] = np.concatenate([s.get('x_untransformed_data', s.get('x_raw', [])) 
            if i==0 
            else s.get('x_untransformed_data', s.get('x_raw', []))[1:] for i, s in enumerate(best_fit_info)])
    
    else:
        original_results = best_fit_info.copy()
        if 'y_data_data' not in original_results: original_results['y_data_data'] = original_results.get('y_raw', [])
        if 'x_untransformed_data' not in original_results: original_results['x_untransformed_data'] = original_results.get('x_raw', original_results.get('x_raw_data', []))

    # 5. Signal Filtering
    original_results['residuals_avg'] = _calculate_moving_average(pd.Series(original_results['residuals']), avg_win_size)
    comparison_df['residuals_avg'] = _calculate_moving_average(comparison_df['residuals'], avg_win_size)
    
    # Step 5: Prepare paths and filenames for saving results
    output_dir = config['main_output_folder']
    num_points_in_fit = len(original_results['y_data_data'])
    base_name_with_pts = f"{config['base_file_name']}_{num_points_in_fit}pts"
    
    
    # Step 6: Identify segment splits for Piecewise models
    segment_splits = config.get('current_data_dict', {}).get('piecewise_splits_T', [])
    if not segment_splits and isinstance(best_fit_info, list):
        segment_splits = [s.get('y_data_data', np.array([0])).max() for s in best_fit_info[:-1]]
     
    # Step 7: Generate and save plots
    plot1_path = os.path.join(output_dir, f"{base_name_with_pts}_raw_residuals_comparison.png")
    _plot_and_save_comparison(original_results, 
                              comparison_df, 
                              'y_data_data',    
                              'T',
                              'residuals', 
                              'residuals', 
                              'Comparison of Raw Residuals', 
                              'Residual, mK', 
                              plot1_path,
                              splits=segment_splits)
    
    plot2_path = os.path.join(output_dir, f"{base_name_with_pts}_avg_residuals_comparison_{avg_win_size}pts.png")
    _plot_and_save_comparison(original_results, 
                              comparison_df, 
                              'y_data_data',    
                              'T',
                              'residuals_avg', 
                              'residuals_avg', 
                              f'Comparison of Averaged Residuals (N={avg_win_size})', 
                              f'Averaged Residual [mK] (N={avg_win_size})', 
                              plot2_path,
                              splits=segment_splits)

    # Step 8: Save the comparison data to CSV files
    _save_comparison_csv(original_results, comparison_df, avg_win_size, output_dir, base_name_with_pts)
    
    print("--- Residual comparison complete ---")