# -*- coding: utf-8 -*-
"""
data_loader.py - Data Ingestion & Standardization Module

This module handles the loading of experimental calibration data from CSV files.
It ensures that varied regional formats (separators and decimal points) are 
automatically resolved and that the resulting DataFrame follows a strict 
schema required by the analysis engines.

Key Features:
- Sensor Agnostic: Automatically maps Voltage (U, V) or Resistance (R) to a 
  standardized internal variable to support PRTs, SPRTs, and Diode sensors.
- Automatic separator detection (comma vs. semicolon).
- Numeric enforcement with NaN removal for corrupted rows.
- Schema standardization: Ensures presence of 'R' (Signal), 'T', 'Rstd', and 'Tstd'.
- Provides fallback uncertainties if source data is unweighted.
"""

import pandas as pd
import os
import logging

def load_data(file_path: str) -> tuple[pd.DataFrame, str]:
    """
    Loads data from a CSV file or generates a sample dataset if the file is not found.

    The function performs automatic detection of delimiters based on the file header.
    It enforces numeric types on temperature (T) and resistance (R) columns, 
    pruning any rows with non-numeric artifacts. It also injects default 
    uncertainty values (Rstd, Tstd) if they are missing from the source file.

    Args:
        file_path (str): Relative or absolute path to the source CSV file.

    Returns:
        tuple[pd.DataFrame, str]: 
            - df: Standardized DataFrame containing [R, T, Rstd, Tstd].
            - absolute_file_path: Resolved absolute path for metadata tracking.

    Raises:
        ValueError: If the required columns 'R' or 'T' are missing.
        FileNotFoundError: If the specified path cannot be resolved.
    """
    absolute_file_path = os.path.abspath(file_path)

    # --- 1. Separator and Decimal Format Detection ---
    # We inspect the header to differentiate between Western (,) and European (;) CSV formats.
    with open(absolute_file_path, 'r', encoding='utf-8') as f:
        header = f.readline()

    if header.count(';') > header.count(','):
        separator = ';'
        decimal_separator = ','
        logging.info("Detected semicolon separator ';'. Using comma ',' as decimal point.")
    else:
        separator = ','
        decimal_separator = '.'
        logging.info("Detected comma separator ','. Using dot '.' as decimal point.")

    # --- 2. Load the CSV with the detected settings ---
    df_raw = pd.read_csv(
        absolute_file_path,
        sep=separator,
        decimal=decimal_separator
    )
    
    logging.info(f"Data successfully loaded from: {absolute_file_path}")
    logging.info(f"Available columns: {df_raw.columns.tolist()}")
    
    # --- 3. Sensor-Agnostic Signal & Temperature Mapping ---
    
    # A) Signal Mapping (R, U, V)
    input_aliases = ['R', 'U', 'V', 'Voltage', 'Resistance']
    found_input_col = next((col for col in input_aliases if col in df_raw.columns), None)
    
    if not found_input_col:
        raise ValueError(f"Missing primary input signal column. Expected one of: {input_aliases}")

    if found_input_col != 'R':
        logging.info(f"Mapping input column '{found_input_col}' to internal variable 'R'")
        df_raw.rename(columns={found_input_col: 'R'}, inplace=True)

    std_aliases = [f"{found_input_col}std", f"{found_input_col}_std", 'Ustd', 'Vstd']
    found_std_col = next((col for col in std_aliases if col in df_raw.columns), None)
    if found_std_col and found_std_col != 'Rstd':
        logging.info(f"Mapping uncertainty column '{found_std_col}' to internal variable 'Rstd'")
        df_raw.rename(columns={found_std_col: 'Rstd'}, inplace=True)

    # B) Temperature Mapping (T, Temp, Temperature)
    temp_aliases = ['T', 'Temp', 'Temperature', 't']
    found_temp_col = next((col for col in temp_aliases if col in df_raw.columns), None)
    
    if not found_temp_col:
        raise ValueError(f"Missing required Temperature column. Expected one of: {temp_aliases}")
        
    if found_temp_col != 'T':
        logging.info(f"Mapping temperature column '{found_temp_col}' to internal variable 'T'")
        df_raw.rename(columns={found_temp_col: 'T'}, inplace=True)
        
    temp_std_aliases = [f"{found_temp_col}std", f"{found_temp_col}_std", 'Tstd']
    found_temp_std_col = next((col for col in temp_std_aliases if col in df_raw.columns), None)
    if found_temp_std_col and found_temp_std_col != 'Tstd':
        logging.info(f"Mapping temperature uncertainty column '{found_temp_std_col}' to internal variable 'Tstd'")
        df_raw.rename(columns={found_temp_std_col: 'Tstd'}, inplace=True)
        
    # --- 4. Data Cleaning & Validation ---
    # Work on a copy to avoid SettingWithCopy warnings       
    required = ['R', 'T']
    if not all(col in df_raw.columns for col in required):
        raise ValueError(f"Missing required columns in CSV: {[c for c in required if c not in df_raw.columns]}")
    
    # Work on a copy to avoid SettingWithCopy warnings
    df = df_raw[required].copy()
    initial_rows = len(df)
    
    # Enforce numeric types; non-numeric values are converted to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows where critical data (R or T) is missing or corrupted
    df.dropna(subset=['R', 'T'], inplace=True)
    final_rows = len(df)

    if final_rows < initial_rows:
        removed_count = initial_rows - final_rows
        message = f"WARNING: Removed {removed_count} rows containing non-numeric or empty 'R'/'T' data."
        print(message)          
        logging.warning(message) 

    # --- 3. Schema Standardization (Uncertainty Fallbacks) ---
    # Analysis engines require Rstd and Tstd for weighted fitting.
    # If missing, we apply standard laboratory defaults.
    
    if 'Rstd' in df_raw.columns:
        df['Rstd'] = df_raw['Rstd']
    else:
        print("INFO: 'Rstd' column not found. Using default value: 1e-6 Ohm. Y errors will be treated as uniform.")
        df['Rstd'] = 1e-6
    
    if 'Tstd' in df_raw.columns:
        df['Tstd'] = df_raw['Tstd']
    else:
        print("INFO: 'Tstd' column not found. Using default value: 0.5 mK. X error bars will not be plotted.")
        df['Tstd'] = 0.0005

    logging.info(f"Successfully loaded {final_rows} points from {os.path.basename(file_path)}")
    return df, absolute_file_path