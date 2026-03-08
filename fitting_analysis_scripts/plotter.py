# -*- coding: utf-8 -*-
"""
plotter.py

This module provides high-fidelity data visualization for thermometer calibration. 
It generates diagnostic plots to evaluate fit quality, residual distribution, 
and model parsimony.

Visual Standards:
-----------------
1. LaTeX Integration: Uses mathematical typesetting for axis labels and units.
2. Silent Generation: Internal helpers (_save_standalone_*) generate plots 
   in the background without interrupting the CLI workflow.
3. Multi-panel Diagnostics: Summary plots combine residuals, goodness-of-fit 
   metrics, and information criteria into a single audit report.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from fitting_analysis_scripts.data_saver import get_global_results_path

def generate_diagnostic_plots(best_result: dict, output_dir: str, file_base_name: str, num_points: int, interactive: bool = False):
    """
    Generates and persists critical diagnostic plots for a best-fit model.
    Includes Standard Residuals, Normal Q-Q, and Studentized Residuals.
    
    This function avoids recalculating statistical metrics by pulling them 
    directly from the 'best_result' dictionary provided by the analyzer.

    Args:
        best_result (dict): The result dictionary for the optimal fit model.
        output_dir (str): The root directory path to save the plot images.
        file_base_name (str): The base name for the output plot files.
        num_points (int): The number of observations in the dataset.
        interactive (bool): If True, displays the plots on screen (non-blocking).
                            If False, generates them silently in the background.
    """
    mode_str = "interactive" if interactive else "background"
    logging.info(f"Generating standalone diagnostic plots ({mode_str} mode)...")
    
    target_dir = get_global_results_path(output_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    # X-axis for plotting physical temperature domain
    x_plot = best_result.get('y_data_data', np.arange(num_points))
    
    # --- 1. Standard Residuals Plot ---
    # Purpose: Check for heteroscedasticity and residual magnitude/trends.
    try:
        fig_res, ax_res = plt.subplots(figsize=(10, 6))
        
        residuals_mK = best_result['residuals'] * 1000
        
        # Robust error bar plotting: checking if uncertainties were actually provided
        if 'std_y_data' in best_result and 'std_x_data' in best_result:
            y_err_mK = best_result['std_y_data'] * 1000
            ax_res.errorbar(x_plot, residuals_mK, 
                            yerr=y_err_mK, xerr=best_result['std_x_data'], 
                            fmt='o', capsize=3, markersize=4, color='purple', label='Residuals', alpha=0.7)
        else:
            # Fallback for data without specified uncertainties
            ax_res.scatter(x_plot, residuals_mK, color='purple', label='Residuals', alpha=0.7)
            
        ax_res.axhline(y=0, color='r', linestyle='--', label='Zero Deviation')
        ax_res.set_title(r'$\mathbf{Standard\ Residuals\ Analysis\, mK}$')
        ax_res.set_xlabel('Temperature, K')
        ax_res.set_ylabel('Residuals, mK')
        ax_res.legend(loc='best')
        ax_res.grid(True, alpha=0.3)
        
        if interactive:
            plt.show(block=False)
            
        path_res = os.path.join(target_dir, f"{file_base_name}_{num_points}pts_standard_residuals.png")
        fig_res.savefig(path_res, dpi=300, bbox_inches='tight')
        plt.close(fig_res)
        logging.info(f"Standard residuals plot saved: {path_res}")
        
    except Exception as e:
        logging.error(f"Failed to generate standard residuals plot: {e}")

    # --- 2. Normal Q-Q Plot ---
    # Purpose: Validate the assumption of normally distributed regression errors.
    try:
        residuals = best_result['residuals']
        fig_qq = plt.figure(figsize=(6, 6))
        ax_qq = fig_qq.add_subplot(111)
        
        sm.qqplot(residuals, line='45', fit=True, ax=ax_qq)
        ax_qq.set_title(r'$\mathbf{Normal\ Q-Q\ Plot}$')
        ax_qq.grid(True, alpha=0.3)
        
        if interactive:
            plt.show(block=False)
            
        path_qq = os.path.join(target_dir, f"{file_base_name}_{num_points}pts_qq_plot.png")
        fig_qq.savefig(path_qq, dpi=300, bbox_inches='tight')
        plt.close(fig_qq)
        logging.info(f"Q-Q plot saved: {path_qq}")
        
    except Exception as e:
        logging.error(f"Failed to generate Q-Q plot: {e}")
        
    # --- 3. Studentized Residuals Plot ---
    # Purpose: Identify leverage-adjusted statistical outliers (|r| > 3).
    try:
        stud_res = best_result.get('studentized_residuals')
        
        if stud_res is not None:
            fig_stud, ax_stud = plt.subplots(figsize=(10, 6))
            
            ax_stud.plot(x_plot, stud_res, 'o', color='teal', alpha=0.7)
            ax_stud.axhline(y=0, color='r', linestyle='--')
            ax_stud.axhline(y=2, color='orange', linestyle=':', label=r'$|r| > 2$ (Unusual)')
            ax_stud.axhline(y=-2, color='orange', linestyle=':')
            ax_stud.axhline(y=3, color='red', linestyle=':', label=r'$|r| > 3$ (Outlier)')
            ax_stud.axhline(y=-3, color='red', linestyle=':')
            
            ax_stud.set_title(r'$\mathbf{Studentized\ Residuals}$')
            ax_stud.set_xlabel(r'$Temperature\ (K)$')
            ax_stud.set_ylabel(r'$Studentized\ Residuals$')
            ax_stud.legend(loc='best')
            ax_stud.grid(True, alpha=0.3)
            
            if interactive:
                plt.show(block=False)
                
            path_stud = os.path.join(target_dir, f"{file_base_name}_{num_points}pts_studentized_residuals.png")
            fig_stud.savefig(path_stud, dpi=300, bbox_inches='tight')
            plt.close(fig_stud)
            logging.info(f"Studentized residuals plot saved: {path_stud}")
        else:
            logging.debug("Skipped Studentized Residuals plot: Data not available in best_result.")
            
    except Exception as e:
        logging.error(f"Failed to generate studentized residuals plot: {e}")

def plot_analysis_results(best_result: dict, data_label: str, all_results_for_current_data: dict,
                          num_points: int, output_dir: str, file_base_name: str, 
                          xlabel: str = 'Polynomial Degree', degree_label: str = 'Degree', 
                          show_plot: bool = True, **kwargs):
    """
    Creates, displays, and saves a 3-in-1 summary plot for a fitting analysis.

    The figure consists of three subplots:
    1. Studentized Residuals vs. Temperature for the best-fit model.
    2. Sum of Absolute Residuals and Reduced Chi-squared vs. Polynomial Degree.
    3. AIC and BIC vs. Polynomial Degree.

    Args:
        best_result (dict): The result dictionary for the best-fit model.
        data_label (str): A descriptive label for the dataset being analyzed.
        all_results_for_current_data (dict): Dictionary with results for all tested degrees.
        num_points (int): The number of points in the dataset.
        output_dir (str): The directory path to save the plot images.
        file_base_name (str): The base name for the output plot files.
        xlabel (str): The label for the x-axis of the comparison plots.
        degree_label (str): The label used in the title for the best fit degree.
        **kwargs: Catches unused arguments like B1_val and polynomial_function for compatibility.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f'Analysis for: {data_label}\n(Dataset contains {num_points} points)', fontsize=16)

    # --- PLOT 1: Studentized Residuals Plot fo Best-Fit ---
    stud_res = best_result.get('studentized_residuals')
    best_degree_val = best_result.get('degree', 'N/A')

    if stud_res is not None:
        axes[0].plot(best_result['y_data_data'], stud_res, 'o', ms=4, label='Studentized Residuals', alpha=0.7)
        axes[0].axhline(y=2, color='orange', ls=':', label='|Residuals| > 2 (Unusual)')
        axes[0].axhline(y=-2, color='orange', ls=':')
        axes[0].axhline(y=3, color='red', ls=':', label='|Residuals| > 3 (Likely Outlier)')
        axes[0].axhline(y=-3, color='red', ls=':')
        axes[0].set_title(f'Studentized Residuals Plot (Best Fit: {degree_label} {best_degree_val})')
        axes[0].set_ylabel('Studentized Residuals')
    else:
        logging.warning("Studentized residuals not found in results. Plotting standard residuals.")
        residuals_mK = best_result['residuals'] * 1000
        y_err_mK = best_result['std_y_data'] * 1000
        axes[0].errorbar(best_result['y_data_data'], residuals_mK,
                         yerr=y_err_mK, xerr=best_result['std_x_data'],
                         fmt='o', capsize=3, ms=4, label='Residuals', alpha=0.7)
        axes[0].set_title(f'Standard Residuals Plot (Best Fit: Degree {best_degree_val})')
        axes[0].set_ylabel('Residuals [mK]')

    axes[0].axhline(y=0, color='r', linestyle='--', label='Zero Residuals')
    axes[0].set_xlabel('Temperature, K'); axes[0].legend(); axes[0].grid(True)

    # --- PLOT 2 & 3: Model Comparison Metrics ---
    degrees = sorted(all_results_for_current_data.keys())
    s_abs_res = [all_results_for_current_data[d]['sum_of_absolute_residuals'] for d in degrees]
    r_chi_sq = [all_results_for_current_data[d]['reduced_chi_squared'] for d in degrees]
    aic = [all_results_for_current_data[d]['aic'] for d in degrees]
    bic = [all_results_for_current_data[d]['bic'] for d in degrees]

    # Goodness-of-Fit (SAR vs Reduced Chi-Squared)
    ax1_twin = axes[1].twinx()
    p1, = axes[1].plot(degrees, s_abs_res, 'o-', c='purple', label='Sum of Absolute Residuals')
    p2, = ax1_twin.plot(degrees, r_chi_sq, 'x-', c='green', label='Reduced Chi-squared')
    axes[1].set_ylabel('Sum of Absolute Residuals', color='purple')
    ax1_twin.set_ylabel('Reduced Chi-squared', color='green')
    axes[1].tick_params(axis='y', labelcolor='purple')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    axes[1].legend(handles=[p1, p2], loc='best')
    axes[1].set_title(f'Goodness-of-Fit vs. {xlabel}'); axes[1].set_xlabel(xlabel); axes[1].grid(True)
    
    # Information Criteria (AIC vs BIC)
    ax2_twin = axes[2].twinx()
    p3, = axes[2].plot(degrees, aic, 's-', c='blue', label='AIC')
    p4, = ax2_twin.plot(degrees, bic, 'd-', c='red', label='BIC')
    axes[2].set_ylabel('AIC', color='blue')
    ax2_twin.set_ylabel('BIC', color='red')
    axes[2].tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    axes[2].legend(handles=[p3, p4], loc='best')
    axes[2].set_title(f'Information Criteria vs. {xlabel}'); axes[2].set_xlabel(xlabel); axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if show_plot:
        plt.show(block=False) 
    
    try:
        target_dir = get_global_results_path(output_dir)
        path = os.path.join(target_dir, f"{file_base_name}_summary_plots.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logging.info(f"Summary plot saved to: {path}")
    except Exception as e:
        logging.error(f"Error saving summary plot: {e}")
        
    generate_diagnostic_plots(best_result, output_dir, file_base_name, num_points)
    plt.close(fig)

def plot_outlier_variability(all_variability_results: dict, plot_title_prefix: str,
                             fixed_degree: int, file_base_name: str, output_dir: str):
    """
    Creates, displays, and saves plots for the outlier variability analysis.

    Visualizes how key fit statistics (Sum of Absolute Residuals, Reduced
    Chi-squared, AIC) change as outliers are progressively removed.

    Args:
        all_variability_results (dict): Dictionary of results for each removal step.
        plot_title_prefix (str): A prefix for the plot titles.
        fixed_degree (int): The polynomial degree used for all fits in the test.
        file_base_name (str): The base name of the original input file.
        output_dir (str): Directory path to save the plot image.
    """
    if not all_variability_results:
        print(f"No results to plot for {plot_title_prefix}.")
        return

    counts = sorted(all_variability_results.keys())
    s_abs_res = [all_variability_results[n]['sum_of_absolute_residuals'] for n in counts]
    r_chi_sq = [all_variability_results[n]['reduced_chi_squared'] for n in counts]
    aic = [all_variability_results[n]['aic'] for n in counts]
    
    x_labels = [f"{c} (idx {all_variability_results[c].get('removed_outlier_indices', ['N/A'])[-1]})" for c in counts]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{plot_title_prefix} - {file_base_name} (Fixed Degree: {fixed_degree})', fontsize=16)

    # Plot 1
    ax0_twin = axes[0].twinx()
    p1, = axes[0].plot(counts, s_abs_res, 'o-', c='purple', label='Sum of Absolute Residuals')
    p2, = ax0_twin.plot(counts, r_chi_sq, 'x-', c='green', label='Reduced Chi-squared')
    axes[0].set_ylabel('Sum of Absolute Residuals', color='purple'); ax0_twin.set_ylabel('Reduced Chi-squared', color='green')
    axes[0].tick_params(axis='y', labelcolor='purple'); ax0_twin.tick_params(axis='y', labelcolor='green')
    axes[0].legend(handles=[p1, p2], loc='best')
    axes[0].set_title('Goodness-of-Fit vs. Outliers Removed')

    # Plot 2
    axes[1].plot(counts, aic, 's-', color='blue', label='AIC')
    axes[1].set_title('AIC vs. Outliers Removed'); axes[1].set_ylabel('AIC'); axes[1].legend()

    for ax in axes:
        ax.set_xlabel('Number of Outliers Removed (Original Index of Last Removed)')
        ax.set_xticks(counts); ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
    try:
        target_dir = get_global_results_path(output_dir)
        path = os.path.join(target_dir, f"{file_base_name}_outlier_variability.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logging.info(f"Outlier variability plot saved to: {path}")
    except Exception as e:
        logging.error(f"Error saving outlier variability plot: {e}")

        
def plot_piecewise_residuals(best_fit_list, current_data, config):
    """
    Generates a unified diagnostic residual plot for piecewise analysis.
    
    This function visualizes the residuals from all calculated segments on a 
    single axis, assigning a unique color to each segment. This allows for 
    immediate visual inspection of boundary continuity and segment-specific 
    heteroscedasticity.

    Args:
        best_fit_list (list): A list of result dictionaries containing 'y_data' 
                              and 'residuals' for each evaluated segment.
        current_data (dict): The active dataset dictionary (used for point counting).
        config (dict): Configuration mapping containing 'main_output_folder'.
    """
    plt.figure(figsize=(10, 6))
    
    total_points = 0
    
    # Iterate through each segment and plot its residuals with a unique color/label
    for i, seg in enumerate(best_fit_list):
        y_vals = seg.get('y_data')
        res_vals = seg.get('residuals')
        
        if y_vals is not None and res_vals is not None:
            plt.scatter(y_vals, res_vals, label=f"Segment {i+1}", alpha=0.7)
            total_points += len(y_vals)
    
    # Establish zero-deviation baseline
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label="Zero Baseline")
    
    # Apply standard metrological formatting
    plt.xlabel("Temperature, K", fontsize=10)
    plt.ylabel("Residuals", fontsize=10)
    plt.title(f"Piecewise Residuals Diagnostic - Total: {total_points} pts", fontsize=12)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Persist the plot to the file system
    output_dir = config.get('main_output_folder')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "piecewise_residuals_diagnostic.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"[*] Piecewise residual plot saved to: {plot_path}")
    
    plt.show()
    plt.close()
        
def plot_piecewise_summary(piecewise_results: list, current_data: dict, all_segment_stats: list, config: dict):
    """
    Generates a high-density diagnostic dashboard for segmented (piecewise) calibrations.
    
    This function produces two distinct figures to evaluate piecewise fits:
    1. Segment-Level Dashboard (N x 3 grid): Evaluates the model selection process 
       within each segment independently (Residuals, AIC/BIC, SAR/Chi-Squared).
    2. Global Stitched Summary (3 x 1 panel): Evaluates the physical continuity 
       of the complete model across all segment boundaries (C0 fit, Global Residuals).

    Args:
        piecewise_results (list): List of best-fit result dictionaries per segment.
        current_data (dict): Global dataset containing raw data and split coordinates.
        all_segment_stats (list): Complete statistics for all tested degrees per segment.
        config (dict): System configuration mapping (paths, model names).
    """

    if not piecewise_results or len(piecewise_results) == 0:
        print("No results to plot.")
        return
    
    n_segments = len(piecewise_results)
    target_dir = config.get('target_report_dir', config['main_output_folder'])
    model_name = config.get('analysis_params', {}).get('model_type', 'Piecewise Model')
    splits = current_data.get('piecewise_splits_T', [])
    
    raw_path = config.get('target_report_dir', config.get('main_output_folder', 'results'))
    path_parts = Path(raw_path).parts
    short_path = os.path.join(*path_parts[-4:]) if len(path_parts) >= 4 else raw_path   
    model_name = config.get('analysis_params', {}).get('model_type', 'Piecewise Model')

    # Generate a distinct color palette for segment differentiation
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_segments))

    # Data collectors for the global stitched diagnostics pane
    all_t, all_res_mk, all_y_err_mk, all_x_err, all_stud_res = [], [], [], [], []

    # =========================================================================
    # 1. SEGMENT-LEVEL DIAGNOSTICS DASHBOARD (N x 3)
    # =========================================================================
    fig_diag, axes_diag = plt.subplots(n_segments, 3, figsize=(18, 5 * n_segments))
    fig_diag.suptitle(f'Piecewise Diagnostics Summary: {model_name}\nTarget: ...{short_path}', fontsize=16)
    
    # Ensure axes_diag is always a 2D array even for a single segment
    if n_segments == 1:
        axes_diag = np.expand_dims(axes_diag, axis=0)

    for i, best_fit in enumerate(piecewise_results):
        ax_res = axes_diag[i, 0]
        ax_aic = axes_diag[i, 1]
        ax_sar = axes_diag[i, 2]
        
        # Extract operational data with safe fallbacks
        t_data = best_fit.get('y_data_data', best_fit.get('y_raw_data'))
        res_raw = best_fit.get('residuals', np.zeros_like(t_data))
        res_mK = res_raw * 1000
        
        y_err = best_fit.get('std_y_data', best_fit.get('std_y', np.zeros_like(t_data)))
        y_err_mK = y_err * 1000
        x_err = best_fit.get('std_x_data', best_fit.get('std_x', np.zeros_like(t_data)))
        stud_res = best_fit.get('studentized_residuals', np.zeros_like(t_data))
        
        # Aggregate data for the global stitched plot
        all_t.extend(t_data); all_res_mk.extend(res_mK)
        all_y_err_mk.extend(y_err_mK); all_x_err.extend(x_err); all_stud_res.extend(stud_res)
        
        # --- COLUMN 1: STANDARD RESIDUALS (mK) ---
        ax_res.errorbar(t_data, res_mK, yerr=y_err_mK, fmt='o', ms=4, color=colors[i], alpha=0.7)
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_title(f"Segment {i+1}: Residuals")
        ax_res.set_xlabel("Temperature, K"); ax_res.set_ylabel("Residuals, mK")
        ax_res.grid(True, alpha=0.3)

        # --- COLUMN 2 & 3: MODEL SELECTION STATS ---
        if i < len(all_segment_stats):
            seg_stats = all_segment_stats[i]
            raw_keys = sorted(seg_stats.keys())
            
            # Determine if the model is Rational (tuple keys) or Polynomial (int keys)
            is_rational = len(raw_keys) > 0 and isinstance(raw_keys[0], tuple)
            
            if is_rational:
                # Rational Logic: Select the best denominator 'm' for each numerator 'n' via AIC
                best_per_n = {}
                for (n, m), st in seg_stats.items():
                    current_aic = st.get('aic', np.inf)
                    if n not in best_per_n:
                        best_per_n[n] = {'m': m, 'stats': st}
                    else:
                        if current_aic < best_per_n[n]['stats'].get('aic', np.inf):
                            best_per_n[n] = {'m': m, 'stats': st}
                    
                degrees = sorted(best_per_n.keys())
                aic_vals = [best_per_n[n]['stats']['aic'] for n in degrees]
                bic_vals = [best_per_n[n]['stats']['bic'] for n in degrees]
                m_labels = [best_per_n[n]['m'] for n in degrees]
                x_label = "Numerator Degree (n)"
            else:
                # Standard Polynomial Logic
                degrees = raw_keys
                aic_vals = [seg_stats[d]['aic'] for d in degrees]
                bic_vals = [seg_stats[d]['bic'] for d in degrees]
                m_labels = None
                x_label = "Degree"

            if degrees:
                # --- Plot: Information Criteria (AIC & BIC) ---
                ax_aic_twin = ax_aic.twinx()
                p1, = ax_aic.plot(degrees, aic_vals, 's-', c='blue', label='AIC', ms=4)
                p2, = ax_aic_twin.plot(degrees, bic_vals, 'd-', c='red', label='BIC', ms=4)
                ax_aic.set_title(f"Segment {i+1}: Information Criteria")
                ax_aic.set_xlabel(x_label)
                ax_aic.set_ylabel("AIC", color='blue'); ax_aic_twin.set_ylabel("BIC", color='red')
                ax_aic.grid(True, alpha=0.3)
                
                # Annotate best 'm' degrees for Rational fits
                if is_rational:
                    y_range = max(aic_vals) - min(aic_vals) if len(aic_vals) > 1 else 1
                    for x, y, m in zip(degrees, aic_vals, m_labels):
                        ax_aic.text(x, y + 0.02 * y_range, f"m={m}", fontsize=8, ha='center', fontweight='bold')

                # --- Plot: Goodness of Fit (SAR & Reduced Chi-Squared) ---
                sar_vals, chi2_vals = [], []
                for d in degrees:
                    st = best_per_n[d]['stats'] if is_rational else seg_stats[d]
                    # Robust extraction of Sum of Absolute Residuals (SAR) and Chi2
                    sar = st.get('sum_of_absolute_residuals', st.get('sum_abs_res', np.sum(np.abs(st.get('residuals', 0)))))
                    chi2 = st.get('reduced_chi_squared', st.get('reduced_chi_sq', 0))
                    sar_vals.append(sar); chi2_vals.append(chi2)

                ax_sar_twin = ax_sar.twinx()
                p3, = ax_sar.plot(degrees, sar_vals, 'o-', c='green', label='Sum Abs Res', ms=4)
                p4, = ax_sar_twin.plot(degrees, chi2_vals, '^-', c='purple', label='Red. $\chi^2$', ms=4)
                ax_sar.set_title(f"Segment {i+1}: Quality Metrics")
                ax_sar.set_xlabel(x_label)
                ax_sar.set_ylabel("Sum Abs Residuals", color='green'); ax_sar_twin.set_ylabel("Reduced $\chi^2$", color='purple')
                ax_sar.grid(True, alpha=0.3)
                
                if is_rational:
                    y_range_sar = max(sar_vals) - min(sar_vals) if len(sar_vals) > 1 else 1
                    for x, y, m in zip(degrees, sar_vals, m_labels):
                        ax_sar.text(x, y + 0.02 * y_range_sar, f"m={m}", fontsize=8, ha='center', fontweight='bold')

    fig_diag.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_diag.subplots_adjust(wspace=0.35, hspace=0.2)
    
    try:
        path_diag = os.path.join(target_dir, f"{model_name.replace(' ', '_')}_piecewise_diagnostics.png")
        fig_diag.savefig(path_diag, dpi=300, bbox_inches='tight')
        logging.info(f"Piecewise diagnostics plot saved to: {path_diag}")
    except Exception as e:
        logging.error(f"Failed to export piecewise diagnostics: {e}")
    plt.show(block=False)
    
    # =========================================================================
    # 2. GLOBAL STITCHED SUMMARY PANEL (3 x 1)
    # =========================================================================
    fig_global, axes_g = plt.subplots(3, 1, figsize=(12, 16), sharex=False)
    fig_global.suptitle(f'Global Piecewise Summary: {model_name}\nTarget: ...{short_path}', fontsize=16)

    ax_c0 = axes_g[0]
    ax_res = axes_g[1]
    ax_stud = axes_g[2]

    # --- PANEL 1: C0 Continuity (Fitted Curve vs Raw Data) ---
    ax_c0_right = ax_c0.twinx()
    ax_c0.scatter(current_data['y'], current_data['x_untransformed'], 
                  color='lightgray', s=15, alpha=0.5, label='All Data (Measured R)')

    for i, best_fit in enumerate(piecewise_results):
        t_meas = best_fit.get('y_data_data', best_fit.get('y_raw_data'))
        r_phys = best_fit.get('x_untransformed_data', best_fit.get('x_untransformed', best_fit.get('x_raw_data')))
        t_fit = best_fit.get('y_fit')

        if t_meas is None or r_phys is None or t_fit is None: continue

        # Sort coordinates to ensure continuous line plotting
        sort_idx = np.argsort(t_meas)
        t_meas_s, r_phys_s, t_fit_s = t_meas[sort_idx], r_phys[sort_idx], t_fit[sort_idx]

        ax_c0.scatter(t_meas_s, r_phys_s, color=colors[i], s=25, edgecolors='k', linewidth=0.5, label=f'Seg {i+1} R')
        ax_c0_right.plot(t_meas_s, t_fit_s, color='red', linestyle='--', linewidth=2.5, label='C0 Fit (Pred T)' if i == 0 else "")

    ax_c0.set_ylabel("Measured Resistance [Ω]", color='blue')
    ax_c0_right.set_ylabel("Predicted Temperature [K]", color='red')
    
    # Merge legends from both Y-axes
    lines_1, labels_1 = ax_c0.get_legend_handles_labels()
    lines_2, labels_2 = ax_c0_right.get_legend_handles_labels()
    ax_c0.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=9)
    ax_c0.grid(True, alpha=0.3)
    ax_c0.set_title("C0 Fit: Resistance & Predicted Temperature")

    # --- PANEL 2: Global Stitched Standard Residuals ---
    ax_res.errorbar(all_t, all_res_mk, yerr=all_y_err_mk, xerr=all_x_err, 
                    fmt='o', color='purple', capsize=3, ms=5, alpha=0.7, label='Stitched Residuals')
    ax_res.axhline(0, color='red', linestyle='--', linewidth=1.5)
    for s_t in splits:
        ax_res.axvline(s_t, color='blue', linestyle=':', linewidth=2, label=f'Split at {s_t} K')
    
    ax_res.set_ylabel("Residuals [mK]")
    ax_res.set_title("Stitched Standard Residuals")
    handles, labels = ax_res.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_res.legend(by_label.values(), by_label.keys(), loc='best')
    ax_res.grid(True)

    # --- PANEL 3: Global Studentized Residuals ---
    ax_stud.plot(all_t, all_stud_res, 'o', color='teal', ms=5, alpha=0.7, label='Stitched Stud. Res.')
    ax_stud.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax_stud.axhline(2, color='orange', linestyle=':', label='|Res| > 2')
    ax_stud.axhline(-2, color='orange', linestyle=':')
    ax_stud.axhline(3, color='red', linestyle=':', label='|Res| > 3')
    ax_stud.axhline(-3, color='red', linestyle=':')
    
    # Mark topological split points
    for s_t in splits:
        ax_stud.axvline(s_t, color='blue', linestyle=':', linewidth=2)
    
    ax_stud.set_xlabel("Temperature, K")
    ax_stud.set_ylabel("Studentized Residuals")
    ax_stud.set_title("Stitched Studentized Residuals")
    handles_s, labels_s = ax_stud.get_legend_handles_labels()
    by_label_s = dict(zip(labels_s, handles_s))
    ax_stud.legend(by_label_s.values(), by_label_s.keys(), loc='best')
    ax_stud.grid(True)

    fig_global.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    try:
        path_global = os.path.join(target_dir, f"{model_name}_global_3panel_summary.png")
        fig_global.savefig(path_global, dpi=300, bbox_inches='tight')
        logging.info(f"Global 3-panel piecewise summary saved to: {path_global}")
    except Exception as e:
        logging.error(f"Failed to export global piecewise summary: {e}")
        
    plt.show(block=False)