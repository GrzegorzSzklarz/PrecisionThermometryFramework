# -*- coding: utf-8 -*-
"""
its90_calculator.py

This module implements the International Temperature Scale of 1990 (ITS-90) for 
Platinum Resistance Thermometers (PRTs). It handles reference functions, 
deviation equations, and the iterative self-consistent correction method.

Logic Flow:
1. Reference Functions (Wr): Standard behavior of an ideal PRT.
2. Deviation Functions (Delta W): Thermometer-specific corrections (a, b, c...).
3. Inverse Functions (T90 from Wr): High-degree polynomial approximations.
4. Self-Consistent Correction: Iterative integration of dW/dT to correct 
   measurements taken slightly off the exact fixed-point temperatures.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================
# --- 1. ITS-90 CONSTANTS & DEFINITIONS ---
# ==========================================

# Coefficients for Wr(T) in the range 13.8033 K to 273.16 K
A_COEFFS = np.array([ 
    -2.13534729, 3.18324720, -1.80143597, 0.71727204, 0.50344027, -0.61899395,
    -0.05332322, 0.28021362, 0.10715224, -0.29302865, 0.04459872, 0.11868632, -0.05248134
])

# Coefficients for T(Wr) inverse function approximation for Wr < 1
B_COEFFS = np.array([
    0.183324722, 0.240975303, 0.209108771, 0.190439972, 0.142648498, 0.077993465,
    0.012475611, -0.032267127, -0.075291522, -0.056470670, 0.076201285, 0.123893204,
    -0.029201193, -0.091173542, 0.001317696, 0.026025526
])

# Coefficients for Wr(T) in the range 273.15 K to 1234.93 K
C_COEFFS = np.array([
    2.78157254, 1.64650916, -0.13714390, -0.00649767, -0.00234444, 0.00511868,
    0.00187982, -0.00204472, -0.00046122, 0.00045724
])

# Coefficients for T(Wr) inverse function approximation for Wr >= 1
D_COEFFS = np.array([
    439.932854, 472.418020, 37.684494, 7.472018, 2.920828, 0.005184, -0.963864,
    -0.188732, 0.191203, 0.049025
])

# --- ITS-90 Reference Data ---
# Official T90 temperatures and reference resistance ratios (Wr) for fixed points.
FIXED_POINTS_DATA = {
    'e-H2':      {'T90': 13.8033,   'Wr': 0.001190068069},
    'e-H2_17K':  {'T90': 17.035,    'Wr': 0.002296459022},
    'e-H2_20K':  {'T90': 20.27,     'Wr': 0.004235355538},
    'Ne':        {'T90': 24.5561,   'Wr': 0.008449736237},
    'O2':        {'T90': 54.3584,   'Wr': 0.091718040322},
    'Ar':        {'T90': 83.8058,   'Wr': 0.215859751998},
    'Hg':        {'T90': 234.3156,  'Wr': 0.844142105150},
    'H2O':       {'T90': 273.16,    'Wr': 1.0},
    'Ga':        {'T90': 302.9146,  'Wr': 1.118138892507},
    'In':        {'T90': 429.7485,  'Wr': 1.609801848113},
    'Sn':        {'T90': 505.0780,  'Wr': 1.892797680730},
    'Zn':        {'T90': 692.6770,  'Wr': 2.568917297742},
    'Al':        {'T90': 933.4730,  'Wr': 3.376008599409},
    'Ag':        {'T90': 1234.93,   'Wr': 4.2864147361}
}

# Definitions of ITS-90 sub-ranges for PRT calibration
SUB_RANGES = {
    '1': {'name': "13.8 K to 273.16 K", 'points': ['e-H2', 'e-H2_17K', 'e-H2_20K', 'Ne', 'O2', 'Ar', 'Hg'], 'terms': ['W-1', '(W-1)**2', 'ln(W)**3', 'ln(W)**4', 'ln(W)**5', 'ln(W)**6', 'ln(W)**7']},
    '2': {'name': "24.5 K to 273.16 K", 'points': ['Ne', 'O2', 'Ar', 'Hg'], 'terms': ['W-1', '(W-1)**2', 'ln(W)', 'ln(W)**2', 'ln(W)**3']},
    '3': {'name': "54.3 K to 273.16 K", 'points': ['O2', 'Ar', 'Hg'], 'terms': ['W-1', '(W-1)**2', 'ln(W)**2']},
    '4': {'name': "83.8 K to 273.16 K", 'points': ['Ar', 'Hg'], 'terms': ['W-1', '(W-1)*ln(W)']},
    '5': {'name': "234.3 K to 302.9 K", 'points': ['Hg', 'Ga'], 'terms': ['W-1', '(W-1)**2']},
    '6': {'name': "273.16 K to 302.9 K", 'points': ['Ga'], 'terms': ['W-1']},
    '7': {'name': "273.16 K to 429.7 K", 'points': ['In'], 'terms': ['W-1']},
    '8': {'name': "273.16 K to 505.1 K", 'points': ['In', 'Sn'], 'terms': ['W-1', '(W-1)**2']},
    '9': {'name': "273.16 K to 692.7 K", 'points': ['Sn', 'Zn'], 'terms': ['W-1', '(W-1)**2']},
    '10': {'name': "273.16 K to 933.5 K", 'points': ['Sn', 'Zn', 'Al'], 'terms': ['W-1', '(W-1)**2', '(W-1)**3']},
}

# ==========================================
# --- 2. CORE MATHEMATICAL ENGINE ---
# ==========================================

def _evaluate_deviation_term(W, term_str, derivative=False):
    """
    Calculates the numerical value of a specific ITS-90 deviation term or its 
    analytical derivative with respect to the resistance ratio W.
    """
    lnW = np.log(np.maximum(W, 1e-15))
    W_m1 = W - 1.0
    W_inv = 1.0 / np.maximum(W, 1e-15)

    if term_str == 'W-1':
        return 1.0 if derivative else W_m1
    if term_str == '(W-1)**2':
        return 2.0 * W_m1 if derivative else W_m1**2
    if term_str == '(W-1)**3':
        return 3.0 * W_m1**2 if derivative else W_m1**3
    if term_str == 'ln(W)':
        return W_inv if derivative else lnW
    if term_str == '(W-1)*ln(W)':
        return lnW + W_m1 * W_inv if derivative else W_m1 * lnW
    if term_str.startswith('ln(W)**'):
        p = int(term_str.split('**')[1])
        if derivative:
            return p * (lnW**(p-1)) * W_inv
        return lnW**p
    return 0.0

def calc_Wr_scalar(T):
    """ITS-90 Reference Function Wr(T)."""
    if T < 273.16:
        x = (np.log(np.maximum(T, 1e-10) / 273.16) + 1.5) / 1.5
        return np.exp(np.polyval(A_COEFFS[::-1], x))
    x = (T - 754.15) / 481.0
    return np.polyval(C_COEFFS[::-1], x)

def calc_dWr_dT_analytical(T):
    """Analytical derivative of the reference function dWr/dT."""
    if T < 273.16:
        x = (np.log(np.maximum(T, 1e-10) / 273.16) + 1.5) / 1.5
        Wr = np.exp(np.polyval(A_COEFFS[::-1], x))
        deriv_coeffs = [i * A_COEFFS[i] for i in range(1, len(A_COEFFS))]
        dP_dx = np.polyval(deriv_coeffs[::-1], x)
        return Wr * dP_dx * (1.0 / (1.5 * T))
    x = (T - 754.15) / 481.0
    deriv_coeffs = [i * C_COEFFS[i] for i in range(1, len(C_COEFFS))]
    dP_dx = np.polyval(deriv_coeffs[::-1], x)
    return dP_dx * (1.0 / 481.0)

def calc_deltaW_val_only(W, coeffs_dict, terms_list):
    """Calculates the deviation value Delta W for a given ratio W."""
    if not coeffs_dict: return 0.0
    return sum(coeffs_dict.get(chr(97 + i), 0.0) * _evaluate_deviation_term(W, term) 
               for i, term in enumerate(terms_list))

def calc_dDeltaW_dW_analytical(W, coeffs_dict, terms_list):
    """Calculates the derivative of the deviation function d(Delta W)/dW."""
    if not coeffs_dict: return 0.0
    return sum(coeffs_dict.get(chr(97 + i), 0.0) * _evaluate_deviation_term(W, term, derivative=True) 
               for i, term in enumerate(terms_list))

def solve_W_from_T(T, coeffs_dict, terms):
    """Solves for W at a target T using the calibrated deviation function."""
    Wr = calc_Wr_scalar(T)
    if not coeffs_dict: return Wr
    W_curr = Wr
    # Fixed iterations (15) for high precision convergence
    for _ in range(15):
        dev = calc_deltaW_val_only(W_curr, coeffs_dict, terms)
        W_curr = Wr + dev
    return W_curr

def dw_dt_analytical(T, coeffs_dict, terms):
    """Full analytical derivative dW/dT, combining reference and deviation functions."""
    dWr_dT = calc_dWr_dT_analytical(T)
    W = solve_W_from_T(T, coeffs_dict, terms)
    dD_dW = calc_dDeltaW_dW_analytical(W, coeffs_dict, terms)
    denominator = 1.0 - dD_dW
    return dWr_dT / denominator if abs(denominator) > 1e-12 else 0.0

def get_integrated_correction(T_meas, T_target, R_tpw, coeffs_dict, terms):
    """Integrates dR/dT between measured and target T to find resistance correction."""
    if abs(T_target - T_meas) < 1e-8: return 0.0
    integral, _ = quad(dw_dt_analytical, T_meas, T_target, args=(coeffs_dict, terms), epsrel=1e-12)
    return integral * R_tpw

# ==========================================
# --- 3. CALIBRATION & INVERSION ---
# ==========================================

def calculate_deviation_coeffs(range_id, measured_readings):
    """
    Computes PRT-specific coefficients (a, b, c...) by solving the linear system
    at fixed points. Uses linalg.solve or least-squares.
    """
    r_tpw = measured_readings['H2O']
    range_info = SUB_RANGES[range_id]
    pts = range_info['points']
    
    W_meas = {p: measured_readings[p] / r_tpw for p in pts}
    W_ref = {p: FIXED_POINTS_DATA[p]['Wr'] for p in pts}

    b_vector = np.array([W_meas[p] - W_ref[p] for p in pts])
    A_matrix = np.array([[_evaluate_deviation_term(W_meas[p], term) for term in range_info['terms']] for p in pts])
    
    try:
        if A_matrix.shape[0] == A_matrix.shape[1]:
            coeffs_vector = np.linalg.solve(A_matrix, b_vector)
        else:
            coeffs_vector = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]
        return {chr(97 + i): val for i, val in enumerate(coeffs_vector)}
    except np.linalg.LinAlgError:
        logging.error(f"Linear algebra error in range {range_id}")
        return None

def calculate_temperature(R_measured, r_tpw, range_id, coeffs):
    """Converts Resistance to T90 [K] using reference functions and deviation."""
    W = R_measured / r_tpw
    range_info = SUB_RANGES[range_id]
    delta_W = calc_deltaW_val_only(W, coeffs, range_info['terms'])

    W_r = W - delta_W
    if W_r < 0: return np.nan

    if W_r < 1:
        term = (W_r**(1/6) - 0.65) / 0.35
        sum_B = sum(B_COEFFS[i] * term**i for i in range(16))
        return 273.16 * sum_B
    
    term = (W_r - 2.64) / 1.64
    sum_D = sum(D_COEFFS[i] * term**i for i in range(10))
    return 273.15 + sum_D

# ==========================================
# --- 4. ADVANCED WORKFLOWS ---
# ==========================================

def perform_self_consistent_correction(measured_data, range_id, num_iterations=15):
    """
    Executes a self-consistent iterative algorithm to correct measured resistances 
    to their nominal ITS-90 fixed-point temperatures (T90).

    This function breaks the circular dependency through an iterative process:
    
    1. Initial Estimate: Assumes zero deviation (ideal PRT behavior) to 
       calculate an initial set of coefficients.
    2. Numerical Integration: Uses the current coefficients to integrate 
       the full analytical derivative dW/dT (reference + deviation) 
       between T_meas and T_target for each fixed point.
    3. Resistance Update: Calculates the resistance correction from the integral 
       of sensor sensitivity.
    4. Refinement: Solves the linear system again using the corrected 
       resistances to update the coefficients.
    5. Convergence: Repeats the process (default: 15 iterations) until 
       the coefficients stabilize, typically reaching sub-microKelvin precision.
    """
    range_info = SUB_RANGES[range_id]
    required_points = range_info['points']
    terms = range_info['terms']
    
    if 'H2O' not in measured_data:
        logging.error("Triple Point of Water (H2O) missing. Aborting ITS-90.")
        return None, None
        
    est_r_tpw = measured_data['H2O']['R']
    current_coeffs = {} 
    corrected_readings = {}

    # Convergence loop
    for i in range(num_iterations):
        # 1. Correct TPW base
        tm_h2o, rm_h2o = measured_data['H2O']['T'], measured_data['H2O']['R']
        dr_h2o = get_integrated_correction(tm_h2o, 273.16, est_r_tpw, current_coeffs, terms)
        est_r_tpw = rm_h2o + dr_h2o
        corrected_readings['H2O'] = est_r_tpw
        
        # 2. Correct other points to their nominal T90
        for pt in required_points:
            if pt not in measured_data: continue
            tm, rm = measured_data[pt]['T'], measured_data[pt]['R']
            tt = FIXED_POINTS_DATA[pt]['T90']
            dr = get_integrated_correction(tm, tt, est_r_tpw, current_coeffs, terms)
            corrected_readings[pt] = rm + dr
            
        # 3. Solve for new coefficients
        current_coeffs = calculate_deviation_coeffs(range_id, corrected_readings)
        if current_coeffs is None: return None, None
            
    return corrected_readings, current_coeffs

def generate_sensitivity_report(range_id, r_tpw, coeffs_dict, output_dir, file_base_name):
    """
    Evaluates and visualizes the thermometer's sensitivity (dR/dT) and normalized 
    sensitivity (dW/dT) across the entire calibrated temperature range. 
    Utilizes the analytical engine to compute the derivative of the reference function 
    and the deviation function simultaneously.

    Returns:
        None: Exports results directly to the file system and displays the plot.
    """
    
    points = SUB_RANGES[range_id]['points'] + ['H2O']
    t_min = min(FIXED_POINTS_DATA[p]['T90'] for p in points)
    t_max = max(FIXED_POINTS_DATA[p]['T90'] for p in points)
    
    t_vals = np.linspace(t_min, t_max, int((t_max - t_min) / 0.02) + 1)
    terms = SUB_RANGES[range_id]['terms']
    
    results = []
    for T in t_vals:
        dw_dt = dw_dt_analytical(T, coeffs_dict, terms)
        results.append([T, r_tpw * dw_dt, dw_dt])
        
    df = pd.DataFrame(results, columns=['Temperature_K', 'dR_dT', 'dW_dT'])
    csv_path = os.path.join(output_dir, f"{file_base_name}_sensitivity.csv")
    
    # Scientific formatting for CSV
    df_fmt = df.copy()
    df_fmt['Temperature_K'] = df_fmt['Temperature_K'].map(lambda x: f"{x:.2f}")
    df_fmt['dR_dT'] = df_fmt['dR_dT'].map(lambda x: f"{x:.6e}")
    df_fmt['dW_dT'] = df_fmt['dW_dT'].map(lambda x: f"{x:.6e}")
    df_fmt.to_csv(csv_path, sep=';', index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, df['dR_dT'], '-', color='b', linewidth=2, label='dR/dT, $\Omega$/K')
    
    for p in points:
        t_pt = FIXED_POINTS_DATA[p]['T90']
        dr_dt_pt = r_tpw * dw_dt_analytical(t_pt, coeffs_dict, terms)
        plt.plot(t_pt, dr_dt_pt, 'ro')
        plt.text(t_pt, dr_dt_pt, f" {p}", fontsize=9, va='bottom')
        
    plt.title(f"ITS-90 Sensitivity: Range {range_id}")
    plt.xlabel("Temperature, K")
    plt.ylabel("Sensitivity dR/dT, $\Omega$/K")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_base_name}_sensitivity_plot.png"), dpi=300)
    plt.show()
    plt.close()
    