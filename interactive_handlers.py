# -*- coding: utf-8 -*-
"""
interactive_handlers.py - User Interface & Workflow Orchestrator

This module acts as the central nervous system of the metrological analysis 
framework. It manages interactive Command Line Interface (CLI) menus, validates 
user inputs, and orchestrates data flow between the core mathematical engines 
(Outlier Analysis, Piecewise Topology, ITS-90 Calibration, and Plotting).

By isolating the interactive loops from the computational backend, this architecture 
ensures high maintainability and enables future transitions to graphical interfaces (GUI).
"""

import os
import logging
import pandas as pd
import numpy as np

import fitting_analysis_scripts.function_defs as function_defs
import fitting_analysis_scripts.outlier_analyzer as outlier_analyzer
import fitting_analysis_scripts.subset_generator as subset_generator
import fitting_analysis_scripts.its90_calculator as its90
import fitting_analysis_scripts.dataset_combiner as dataset_combiner
import fitting_analysis_scripts.residual_comparator as residual_comparator
import fitting_analysis_scripts.data_saver as data_saver
import fitting_analysis_scripts.plotter as plotter

# =============================================================================
# --- [SECTION 1: GENERAL HELPER FUNCTIONS & I/O] ---
# =============================================================================

def get_data_folder_path(default_path: str) -> str:
    """
    Prompts the user to define the working directory containing raw calibration data.
    Runs a validation loop to ensure the specified path exists on the local filesystem.
    
    Args:
        default_path (str): The fallback directory (usually 'data') used if the 
                            user submits an empty input.
                            
    Returns:
        str: A validated, absolute path to the target data directory.
    """
    while True:
        print(f"\nEnter path to the data folder (leave blank for: {default_path}):")
        user_input = input().strip()
        
        if not user_input:
            target_path = default_path
        else:
            target_path = user_input

        if os.path.isdir(target_path):
            return os.path.abspath(target_path)
        else:
            print(f"Error: The path '{target_path}' is not a valid directory. Please try again.")

def select_file_from_list(file_list: list) -> str or None:
    """
    Renders an indexed, interactive menu of discovered CSV files.

    Args:
        file_list (list[str]): A list of file paths to display.

    Returns:
        str or None: The full path of the selected file, or None if the user chooses to exit.
    """
    print("\n--- Select a CSV file to analyze ---")
    for i, file_path in enumerate(file_list):
        print(f"{i + 1}. {os.path.basename(file_path)}")
    print("0. Exit Program")
    
    while True:
        try:
            prompt = f"Enter the number of the file (1-{len(file_list)}) or 0 to exit: "
            choice_str = input(prompt).strip()
            
            if choice_str == '0':
                return None
            
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(file_list):
                selected_file = file_list[choice_idx]
                logging.info(f"Selected file: {os.path.basename(selected_file)}")
                return selected_file
            else:
                print("Error: Number is out of range.")
        except (ValueError, IndexError):
            print("Error: Invalid input. Please enter a number from the list.")


def sanitize_foldername(name: str) -> str:
    """
    Normalizes a string for safe usage in filesystem directory names.
    Replaces mathematical operators and spaces with alphanumeric equivalents.
    
    Args:
        name (str): The raw input string (e.g., 'A + B').
        
    Returns:
        str: A sanitized, filesystem-safe string (e.g., 'A_Plus_B').
    """
    name = name.replace(' ', '_').replace('+', 'Plus')
    return "".join(c for c in name if c.isalnum() or c == '_')


def get_float_input(prompt: str) -> float:
    """
    Safely captures floating-point user input, automatically handling 
    regional comma-to-dot decimal separator conversions.
    """
    
    while True:
        try:
            return float(input(prompt).replace(',', '.'))
        except ValueError:
            print("[Error]: Please enter a valid number.")
            
# =============================================================================
# --- [SECTION 2: MAIN MENU & MODEL SELECTION] ---
# =============================================================================

def select_fitting_function(default_function_name: str) -> str:
    """
    Renders the primary algorithmic selection menu. 
    Allows the user to choose between standard mathematical models 
    or the specialized ITS-90 calibration routine.
    
    Args:
        default_function_name (str): The model pre-selected upon pressing Enter.
        
    Returns:
        str: The registered name of the mathematical model, or the specific 
             "ITS-90_CALIBRATION" execution flag.
    """
    function_options = function_defs.list_fitting_functions()
    function_names = list(function_options.keys())
    try:
        default_idx = function_names.index(default_function_name)
    except ValueError:
        default_idx = 0
        
    scaling_descriptions = {
        'none': "(uses raw X)",
        'linear': "(linear scaling)",
        'log': "(log scaling)"
    }
    
    print("\n--- Select Fitting Function or Calibration Mode ---")
    print("0. ITS-90 SPRT Calibration")
    for i, name in enumerate(function_names):
        info = function_options[name]
        scaling_type = info.get('scaling_type', 'none')
        scaling_info = scaling_descriptions.get(scaling_type, "")
        print(f"{i + 1}. {name} {scaling_info}")
    
    prompt = f"Enter number (or press Enter for default: {function_names[default_idx]}): "
    choice_str = input(prompt).strip()
    
    if not choice_str:
        return function_names[default_idx]

    try:
        choice_idx = int(choice_str) - 1
        if choice_idx == -1:
            return "ITS-90_CALIBRATION"
        if 0 <= choice_idx < len(function_names):
            return function_names[choice_idx]
    except ValueError:
        pass

    print(f"Invalid input. Using default function: '{function_names[default_idx]}'")
    return function_names[default_idx]

# =============================================================================
# --- [SECTION 3: CORE WORKFLOW LOOPS] ---
# =============================================================================

def run_fit_analysis_loop(current_data: dict, analysis_results: dict, config: dict) -> tuple:
    """
    The central interactive analysis loop. Handles outlier detection, topological 
    segmentation (Piecewise), cross-validation, and dataset reduction.
    
    Args:
        current_data (dict): The active dataset and physical metadata.
        analysis_results (dict): Dictionary containing regression results ('full', 'best_fit').
        config (dict): Global session configuration.
        
    Returns:
        tuple: (updated_data, updated_results, should_exit_program, should_restart_analysis)
    """
    while True:
        print("\n--- Analysis Menu ---")
        print(f"Current dataset: '{current_data['label']}' ({current_data['num_points']} points)")
        print("  1. Z-score Thresholding")
        print("  2. Interquartile Range (IQR)")
        print("  3. Studentized Residuals (Recommended)")
        print("  4. Generate Residuals Plot & Heteroskedasticity Tests")
        print("  5. Cross-Validation (Compare against secondary dataset)")
        print("  6. Fuse dataset with external CSV (Merge & Combine)") 
        print("  7. Sub-divide dataset into thermal segments (Piecewise)")
        print("  8. Revert to Global Analysis (Remove piecewise knots)")
        print("  9. Random Sub-setting (Data reduction generator)")             
        print("  0. Return to Main Menu (Change dataset/model)")
        print("  q. Terminate Application")
        
        user_input = input("Select an option: ").strip().lower()
        
        if user_input == 'q':
            print("\n[!] Closing program. Goodbye!")
            return current_data, analysis_results, True, False 
        
        if user_input == '8':
            print("\n[!] Returning to Global Analysis...")
            return current_data, analysis_results, False, True

        elif user_input == '0':
            print("Exiting to main menu...")
            return current_data, analysis_results, False, False

        try:
            outlier_choice = int(user_input)
            if outlier_choice not in [1, 2, 3, 4, 5, 6, 7, 9]: 
                raise ValueError
        except ValueError:
            print("Invalid choice. Please select a valid option (1-9, 0, or q).")
            continue
        
        # --- Residual Reconstruction (Critical for Piecewise Models) ---
        best_fit_info = analysis_results.get('best_fit')
        all_residuals = None

        if best_fit_info is not None:
            if isinstance(best_fit_info, list):
                # Topological Stitching: Average the residuals exactly at the knot boundary 
                # to maintain array length synchrony with the physical X/Y arrays.
                all_residuals = []
                for i, seg in enumerate(best_fit_info):
                    seg_res = list(seg.get('residuals', []))
                    if i > 0 and len(seg_res) > 0:
                        all_residuals[-1] = (all_residuals[-1] + seg_res[0]) / 2.0
                        all_residuals.extend(seg_res[1:])
                    else:
                        all_residuals.extend(seg_res)
                all_residuals = np.array(all_residuals)
            else:
                all_residuals = best_fit_info.get('residuals')

        if all_residuals is None and outlier_choice in [1, 2, 3, 4, 5]:
            print("Cannot perform operation: analysis results are not available.")
            continue
        
        outlier_indices_found = None

        # --- OUTLIER DETECTION SUITE --- 
        # --- CASE 1: Z-score ---
        if outlier_choice == 1:
            try:
                threshold_str = input("Enter Z-score threshold (default: 2.0): ").strip()
                threshold = float(threshold_str.replace(',', '.') or 2.0)
                outlier_indices_found = outlier_analyzer.analyze_z_score(all_residuals, current_data['y'], threshold)
            except ValueError:
                print("[!] Invalid input."); continue

        # --- CASE 2: IQR ---
        elif outlier_choice == 2:
            try:
                factor_str = input("Enter IQR factor (default: 1.5): ").strip()
                factor = float(factor_str.replace(',', '.') or 1.5)
                outlier_indices_found = outlier_analyzer.analyze_iqr(all_residuals, current_data['y'], factor)
            except ValueError:
                print("[!] Invalid input."); continue

        # --- CASE 3: Studentized Residuals ---
        elif outlier_choice == 3:
            try:
                threshold_str = input("Enter Studentized Residual threshold (default: 3.0): ").strip()
                threshold = float(threshold_str.replace(',', '.') or 2.0)
                # Requires leverage (Hat Matrix) calculation; relies strictly on best fit object
                outlier_indices_found = outlier_analyzer.analyze_studentized_residuals(
                    current_data['x'], 
                    current_data['y'], 
                    analysis_results['best_fit'], 
                    threshold
                )
            except ValueError:
                print("[!] Invalid input."); continue
                
        # --- DIAGNOSTICS & PLOTTING ---
        # --- CASE 4: Diagnostic Plots ---
        elif outlier_choice == 4:
            if isinstance(best_fit_info, list):
                print("\n--- Generating Piecewise Diagnostic Residual Plot ---")
                plotter.plot_piecewise_residuals(best_fit_info, current_data, config)
                continue

            all_fits = analysis_results.get('full')
            if not all_fits:
                print("Cannot perform operation: dictionary of all fits is not available.")
                continue

            first_key = next(iter(all_fits.keys()))
            if isinstance(first_key, tuple):
                # Rational Function handling (n, m)
                try:
                    n_choice = int(input("Enter numerator degree (n): ").strip())
                    m_choice = int(input("Enter denominator degree (m): ").strip())
                    selected_fit_info = all_fits.get((n_choice, m_choice))
                    file_suffix = f"_n{n_choice}_m{m_choice}"
                except ValueError: continue
            else: 
                # Polynomial handling
                try:
                    degree_choice = int(input("Enter polynomial degree: ").strip())
                    selected_fit_info = all_fits.get(degree_choice)
                    file_suffix = f"_deg{degree_choice}"
                except ValueError: continue
            
            if selected_fit_info:
                outlier_analyzer.visualize_and_test_residuals(
                    residuals=selected_fit_info['residuals'], 
                    x_raw=selected_fit_info.get('x_raw_data', 
                    current_data['x']),
                    x_for_plot=current_data['x'],
                    y_raw=current_data['y'], 
                    best_fit_info=selected_fit_info,
                    output_dir=config['main_output_folder'],
                    file_base_name=config['base_file_name'] + file_suffix
                )
            continue
            
        # --- CASE 5: Cross-validation ---
        elif outlier_choice == 5:
            print("\n--- Comparing Residuals with Another Dataset ---")
            try:
                residual_comparator.run_comparison(best_fit_info, config)
            except ImportError:
                print("[!] Error: residual_comparator module not found.")
            continue
        
        # --- CASE 6 & 7: Topological Segmentation (piecewise) ---
        elif outlier_choice in [6, 7]:
            if outlier_choice == 6:
                current_data = dataset_combiner.combine_with_secondary_dataset(current_data, config)
                p_mode = 'combined'
            else:
                current_data = dataset_combiner.prepare_piecewise_division(current_data, config)
                p_mode = 'divided'
                
            config['piecewise_mode'] = p_mode

            current_n = current_data['num_points']
            current_parent = config.get('main_output_folder', '')
            folder_name = os.path.basename(current_parent)
            
            # Prevent recursive folder nesting
            if folder_name.startswith('divided_') or folder_name.startswith('combined_'):
                current_parent = os.path.dirname(current_parent)
                
            pw_output_folder = os.path.join(current_parent, f"{p_mode}_{current_n}pts")
            local_config = config.copy()
            local_config['main_output_folder'] = pw_output_folder
            local_config['piecewise_mode'] = p_mode
            os.makedirs(pw_output_folder, exist_ok=True)
            
            pw_best_results, pw_all_stats = dataset_combiner.run_intelligent_piecewise_analysis(current_data, local_config)
            if pw_best_results:
                analysis_results['full'] = pw_all_stats
                analysis_results['best_fit'] = pw_best_results 
                plotter.plot_piecewise_summary(pw_best_results, current_data, pw_all_stats, local_config)
                data_saver.save_piecewise_results(pw_best_results, current_data, local_config)
            continue
        
        # --- CASE 9: Subset Generation ---
        elif outlier_choice == 9:
            print("\n--- Entering Subset Generator ---")
            try:
                subset_generator.run_subset_analysis_loop(current_data, analysis_results, config)
            except ImportError:
                print("[!] Error: subset_generator module not found.")
            continue

        # --- OUTLIER REMOVAL EXECUTION ---
        if outlier_indices_found is not None and len(outlier_indices_found) > 0:
            confirm = input("Do you want to remove these points from the dataset? (y/n, default: n): ").strip().lower()
            
            if confirm == 'y':
                print(f"Removing {len(outlier_indices_found)} points...")
                mask = np.ones(len(current_data['y']), dtype=bool)
                mask[outlier_indices_found] = False
                
                # Update underlying physical arrays
                current_data['x'] = current_data['x'][mask]
                current_data['y'] = current_data['y'][mask]
                current_data['std_x'] = current_data['std_x'][mask]
                current_data['std_y'] = current_data['std_y'][mask]
                current_data['x_untransformed'] = current_data['x_untransformed'][mask]
                current_data['num_points'] = len(current_data['y'])
                base_name = config.get('base_file_name', 'dataset')
                current_data['label'] = f"{base_name}_{current_data['num_points']}pts"
                
                # Check if we are operating in a Piecewise environment
                p_mode = config.get('piecewise_mode')
                if p_mode in ['divided', 'combined']:
                    current_n = current_data['num_points']
                    current_parent = config.get('main_output_folder', '')
                    folder_name = os.path.basename(current_parent)
                    if folder_name.startswith('divided_') or folder_name.startswith('combined_'):
                        current_parent = os.path.dirname(current_parent)
                    
                    pw_output_folder = os.path.join(current_parent, f"{p_mode}_{current_n}pts")
                    local_config = config.copy()
                    local_config['main_output_folder'] = pw_output_folder
                    os.makedirs(pw_output_folder, exist_ok=True)
                    
                    pw_best, pw_all = dataset_combiner.run_intelligent_piecewise_analysis(current_data, local_config)
                    analysis_results['full'] = pw_all
                    analysis_results['best_fit'] = pw_best
                    
                    plotter.plot_piecewise_summary(pw_best, current_data, pw_all, local_config)
                    data_saver.save_piecewise_results(pw_best, current_data, local_config)
                    print(f"[*] Dataset updated. Results in: {pw_output_folder}")
                    continue
                else:
                    #Trigger a full re-analysis in the parent module (Global/Rational)
                    return current_data, analysis_results, False, True
            else:
                continue

    return current_data, analysis_results, False

def handle_x_transformation(current_data: dict) -> tuple:
    """
    Renders an interactive menu for defining the fundamental metrological 
    transformation applied to the physical Resistance (Ohms) data prior to fitting.
    Applies standard error propagation physics to calculate the new standard deviations.

    Args:
        current_data (dict): The active dataset containing physical arrays ('x', 'std_x').

    Returns:
        tuple: (Transformed_X_Array, Transformed_StdX_Array, String_Label, Metadata_Dict)
    """

    print("\n--- X-axis Transformation ---")
    print(" Select the physical representation for the independent variable:")
    print("0. Use raw R (default)")
    print("1. Use W = R / R(TPW)")
    print("2. Use ln(W)")
    print("3. Use W_Ne = R / R(TPNe)")
    print("4. Use ln(W_Ne)")
    print("5. Use W_Ar = R / R(TPAr)")
    print("6. Use ln(W_Ar)")
    print("7. Use ln(R)")
    
    choice = input("Select an option (0-7, or Enter for default): ").strip()
    original_x, original_std_x = current_data['x'], current_data['std_x']
    
    try:
        if choice in ['1', '2']:
            r_tpw = get_float_input("Enter resistance at Triple Point of Water (R_tpw) [Ω]: ")
            if r_tpw <= 0: raise ValueError("R(TPW) must be positive.")
            W = original_x / r_tpw
            std_W = original_std_x / r_tpw
            if choice == '1':
                return W, std_W, "W_TPW", {'type': 'W_TPW', 'r_ref': r_tpw}
            else:
                return np.log(W), (1 / W) * std_W, "ln_W", {'type': 'ln_W', 'r_ref': r_tpw}

        elif choice in ['3', '4']:
            r_tpne = get_float_input("Enter resistance at Triple Point of Neon (R_tpne) [Ω]: ")
            if r_tpne <= 0: raise ValueError("R(TPNe) must be positive.")
            W_Ne = original_x / r_tpne
            std_W_Ne = original_std_x / r_tpne
            if choice == '3':
                return W_Ne, std_W_Ne, "W_Ne", {'type': 'W_Ne', 'r_ref': r_tpne}
            else:
                return np.log(W_Ne), (1 / W_Ne) * std_W_Ne, "ln_W_Ne", {'type': 'ln_W_Ne', 'r_ref': r_tpne}

        elif choice in ['5', '6']:
            r_tpar = get_float_input("Enter resistance at Triple Point of Argon (R_tpAr) [Ω]: ")
            if r_tpar <= 0: raise ValueError("R(TPAr) must be positive.")
            W_Ar = original_x / r_tpar
            std_W_Ar = original_std_x / r_tpar
            if choice == '5':
                return W_Ar, std_W_Ar, "W_Ar", {'type': 'W_Ar', 'r_ref': r_tpar}
            else:
                return np.log(W_Ar), (1 / W_Ar) * std_W_Ar, "ln_W_Ar", {'type': 'ln_W_Ar', 'r_ref': r_tpar}
        
        elif choice == '7':
            return np.log(original_x), (1 / original_x) * original_std_x, "ln_R", {'type': 'ln_R'}
        
        else: 
            return None, None, "raw_R", {'type': 'raw_R'}

    except Exception as e:
        logging.error(f"Transformation engine failed: {e}")
        return None, None, "raw_R", {'type': 'raw_R'}
        
def handle_its90_calibration(current_data: dict, config: dict):
    """
    Orchestrates the high-precision ITS-90 Standard Platinum Resistance Thermometer (SPRT) 
    calibration workflow.
    
    Metrological Logic:
    1. Cross-references measured dataset against internationally defined fixed points (T90).
    2. Ascertains if empirical measurements exactly match the theoretical T90 coordinates.
    3. If exact: Dispatches analytical direct linear solvers to find deviation coefficients.
    4. If deviating: Deploys self-consistent iterative integration to dynamically correct R(T).
    5. Exports computed coefficient limits and derivative sensitivity maps (dR/dT).
    """
       
    logging.info("Initializing ITS-90 Precision Calibration session.")

    # --- UI: RANGE SELECTION ---
    print("\n" + "="*50)
    print("   ITS-90 PRECISION CALIBRATION WORKFLOW")
    print("="*50)

    print("\nAvailable ITS-90 Calibration Sub-ranges:")
    for key in sorted(its90.SUB_RANGES.keys(), key=int):
        print(f"  {key}. {its90.SUB_RANGES[key]['name']}")
    
    range_id = input("\n Define target range [ID]: ").strip()
    if range_id not in its90.SUB_RANGES:
        logging.warning(f"Unrecognized sub-range requested: {range_id}")
        print(" [Error] Invalid configuration. Aborting sequence.")
        return
        
    range_info = its90.SUB_RANGES[range_id]
    points_to_process = ['H2O'] + range_info['points']
    
    # --- TECHNICAL STEP: GEOSPATIAL MAPPING ---
    logging.info(f"Scanning input data for range: {range_info['name']}")
    measured_readings = {}
    tolerance_k = 2.0 
    all_points_exact = True 
    
    temp_array = current_data['y']
    res_array = current_data['x']

    for pt in points_to_process:
        target_t = its90.FIXED_POINTS_DATA[pt]['T90']
        diffs = np.abs(temp_array - target_t)
        idx_min = np.argmin(diffs)
        
        # Determine proximity to theoretical fixed point
        if diffs[idx_min] <= tolerance_k:
            t_meas = temp_array[idx_min]
            measured_readings[pt] = {'T': t_meas, 'R': res_array[idx_min]}
            logging.info(f"Fixed point {pt} identified: T_meas = {t_meas:.4f} K")
            
            if abs(t_meas - target_t) > 1e-6:
                all_points_exact = False
        else:
            # Override mapping with manual coordinates
            logging.warning(f"Data for node '{pt}' (~{target_t:.4f} K) absent. Requesting manual override.")
            print(f"\n [!] Theoretical node missing: {pt}")
            try:
                m_t = float(input(f"    Enter measured Temperature for {pt}, K: "))
                m_r = float(input(f"    Enter measured Resistance for {pt}, Ohm: "))
                measured_readings[pt] = {'T': m_t, 'R': m_r}
                if abs(m_t - target_t) > 1e-6:
                    all_points_exact = False
            except ValueError:
                logging.error("Manual override aborted due to non-numeric string.")
                print(" [Fatal] Invalid format. Terminating calibration.")
                return

    # --- TECHNICAL STEP: ITS-90 ALGORITHM EXECUTION ---
    try:
        if all_points_exact:
            logging.info("Executing Direct Analytical Method (Exact T90).")
            r_only_dict = {k: v['R'] for k, v in measured_readings.items()}
            final_coeffs_raw = its90.calculate_deviation_coeffs(range_id, r_only_dict)
            r_tpw = r_only_dict['H2O']
        else:
            logging.info("Executing Self-Consistent Iterative Integration (Deviating temperatures).")
            final_readings, final_coeffs_raw = its90.perform_self_consistent_correction(
                measured_readings, range_id
            )
            r_tpw = final_readings['H2O']

        if final_coeffs_raw is None:
            logging.error("Calibration engine failed to converge or encountered a singular matrix.")
            print("\n[ERROR] Calibration failed: Numerical instability.")
            return

        # --- DATA NORMALIZATION & MAPPING ---
        # Map raw internal dictionary keys to ITS-90 standard nomenclature
        mapped_coeffs = {}
        coeff_keys = sorted(final_coeffs_raw.keys()) 
        for i, key in enumerate(coeff_keys):
            name = 'a' if i == 0 else ('b' if i == 1 else f'c{i-1}')
            mapped_coeffs[name] = final_coeffs_raw[key]
            
        # --- UI: DISPLAY RESULTS ---
        print("\n" + "-"*30)
        print("   CALIBRATION RESULTS")
        print("-"*30)
        print(f"  R_tpw (R0): {r_tpw:.12f} Ohm")
        for key, val in mapped_coeffs.items():
            print(f"  {key:3}: {val:+.12e}")
        print("-"*30)

        # --- TECHNICAL STEP: SYSTEM EXPORT ---
        output_dir = config['main_output_folder']
        file_base = f"{config['base_file_name']}_range_{range_id}"

        coeff_path = os.path.join(output_dir, f"{file_base}_its90_calculated_coeffs.csv")
        save_df_list = [{'parameter': 'R_tpw (R0)', 'value': f"{r_tpw:.12f}", 'unit': 'Ohm'}]
        for k, v in mapped_coeffs.items():
            save_df_list.append({'parameter': k, 'value': f"{v:+.12e}", 'unit': '-'})
            
        pd.DataFrame(save_df_list).to_csv(coeff_path, sep=';', index=False)
        logging.info(f"ITS-90 coefficients exported to: {coeff_path}")

        # Dispatch automated sensitivity report rendering
        its90.generate_sensitivity_report(
            range_id, r_tpw, final_coeffs_raw, output_dir, file_base
        )
        
        logging.info("ITS-90 PRT Calibration procedure completed successfully.")
        print("\n[+] Calibration finished. Reports generated in results folder.")

    except Exception as e:
        logging.error(f"Critical failure during ITS-90 calculation: {e}")
        print(" A critical error occurred. Check analysis_log.txt for details.")
    
    return