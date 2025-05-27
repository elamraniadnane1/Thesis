import pandas as pd
import re
import os
from pathlib import Path

def parse_administrative_level(code, name):
    """Determine the administrative level based on code pattern and name"""
    # Remove any "pm" values and clean the name
    if pd.isna(name) or name == 'pm':
        return None, None
    
    name = str(name).strip()
    code = str(code).strip()
    
    # Check for Region (code format: XX.)
    if re.match(r'^\d{2}\.$', code) and 'Région:' in name:
        return 'region', name.replace('Région:', '').strip()
    
    # Check for Province/Prefecture (code format: XX.XXX.)
    elif re.match(r'^\d{2}\.\d{3}\.$', code):
        if 'Province:' in name:
            return 'province', name.replace('Province:', '').strip()
        elif 'Préfecture:' in name:
            return 'prefecture', name.replace('Préfecture:', '').strip()
    
    # Check for Municipality
    elif '(Mun.)' in name:
        return 'municipality', name.replace('(Mun.)', '').strip()
    
    # Check for Arrondissement
    elif '(Arrond.)' in name:
        return 'arrondissement', name.replace('(Arrond.)', '').strip()
    
    # Check for Cercle (administrative circle)
    elif 'Cercle :' in name or 'Cercle:' in name:
        return 'cercle', name.replace('Cercle :', '').replace('Cercle:', '').strip()
    
    # Check for "Dont Centre" (sub-centers)
    elif 'Dont Centre:' in name:
        return 'centre', name.replace('Dont Centre:', '').strip()
    
    # Everything else is a commune
    elif re.match(r'^\d{2}\.\d{3}\.\d{2}\.\d{2}\.?$', code):
        return 'commune', name
    
    return None, None

def extract_parent_codes(code):
    """Extract parent administrative codes from a full code"""
    parts = code.split('.')
    
    parent_codes = {
        'region_code': None,
        'province_code': None,
        'cercle_code': None
    }
    
    if len(parts) >= 1 and parts[0]:
        parent_codes['region_code'] = parts[0] + '.'
    
    if len(parts) >= 2 and parts[1]:
        parent_codes['province_code'] = f"{parts[0]}.{parts[1]}."
    
    if len(parts) >= 3 and parts[2]:
        parent_codes['cercle_code'] = f"{parts[0]}.{parts[1]}.{parts[2]}."
    
    return parent_codes

def process_census_data(file_path, output_dir):
    """Process the census Excel file and create separate CSV files"""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the Excel file
    print(f"Reading Excel file: {file_path}")
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Print column names to understand the structure
    print("Columns found:", df.columns.tolist())
    
    # Initialize dictionaries to store data by type
    data_by_type = {
        'region': [],
        'province': [],
        'prefecture': [],
        'municipality': [],
        'arrondissement': [],
        'commune': [],
        'cercle': [],
        'centre': []
    }
    
    # Store parent information for lookups
    regions = {}
    provinces = {}
    cercles = {}
    
    # Process each row
    for idx, row in df.iterrows():
        # Get the code and name columns (adjust based on actual column names)
        # Assuming first column is code and second is name
        code = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ''
        name_fr = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ''
        
        # Skip empty rows
        if not code or code == 'nan':
            continue
        
        # Get numeric data
        menages = row.iloc[2] if len(row) > 2 and not pd.isna(row.iloc[2]) and row.iloc[2] != 'pm' else None
        population = row.iloc[3] if len(row) > 3 and not pd.isna(row.iloc[3]) and row.iloc[3] != 'pm' else None
        etrangers = row.iloc[4] if len(row) > 4 and not pd.isna(row.iloc[4]) and row.iloc[4] != 'pm' else None
        marocains = row.iloc[5] if len(row) > 5 and not pd.isna(row.iloc[5]) and row.iloc[5] != 'pm' else None
        name_ar = str(row.iloc[6]) if len(row) > 6 and not pd.isna(row.iloc[6]) else ''
        
        # Determine administrative level
        level, clean_name = parse_administrative_level(code, name_fr)
        
        if not level:
            continue
        
        # Extract parent codes
        parent_codes = extract_parent_codes(code)
        
        # Create data entry
        entry = {
            'code': code,
            'name_fr': clean_name,
            'name_ar': name_ar,
            'menages': menages,
            'population': population,
            'etrangers': etrangers,
            'marocains': marocains
        }
        
        # Add parent information based on level
        if level == 'region':
            regions[code] = clean_name
            
        elif level in ['province', 'prefecture']:
            entry['region_code'] = parent_codes['region_code']
            entry['region_name'] = regions.get(parent_codes['region_code'], '')
            provinces[code] = clean_name
            
        elif level == 'cercle':
            entry['region_code'] = parent_codes['region_code']
            entry['region_name'] = regions.get(parent_codes['region_code'], '')
            entry['province_code'] = parent_codes['province_code']
            entry['province_name'] = provinces.get(parent_codes['province_code'], '')
            cercles[code] = clean_name
            
        elif level in ['municipality', 'arrondissement', 'commune', 'centre']:
            entry['region_code'] = parent_codes['region_code']
            entry['region_name'] = regions.get(parent_codes['region_code'], '')
            entry['province_code'] = parent_codes['province_code']
            entry['province_name'] = provinces.get(parent_codes['province_code'], '')
            entry['cercle_code'] = parent_codes['cercle_code']
            entry['cercle_name'] = cercles.get(parent_codes['cercle_code'], '')
        
        # Add to appropriate list
        data_by_type[level].append(entry)
    
    # Create CSV files for each type
    for data_type, data_list in data_by_type.items():
        if data_list:  # Only create file if there's data
            output_file = os.path.join(output_dir, f'{data_type}s.csv')
            df_output = pd.DataFrame(data_list)
            
            # Reorder columns for better readability
            base_cols = ['code', 'name_fr', 'name_ar']
            parent_cols = [col for col in df_output.columns if col.endswith('_code') or col.endswith('_name')]
            data_cols = ['menages', 'population', 'etrangers', 'marocains']
            
            # Filter existing columns
            ordered_cols = []
            for col in base_cols + parent_cols + data_cols:
                if col in df_output.columns:
                    ordered_cols.append(col)
            
            df_output = df_output[ordered_cols]
            df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Created {output_file} with {len(data_list)} records")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Morocco Census Data Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        for data_type, data_list in data_by_type.items():
            if data_list:
                f.write(f"{data_type.capitalize()}s: {len(data_list)} records\n")
    
    print(f"\nProcessing complete! Files saved to: {output_dir}")

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = r"C:\Users\LENOVO\OneDrive\Bureau\POPULATION LÉGALE DES RÉGIONS, PROVINCES, PRÉFECTURES, MUNICIPALITÉS, ARRONDISSEMENTS ET COMMUNES DU ROYAUME D'APRÈS LES RÉSULTATS DU RGPH 2014 (16 Régions).xlsx"
    output_directory = r"C:\Users\LENOVO\OneDrive\Bureau\Morocco_Census_CSV"
    
    try:
        process_census_data(input_file, output_directory)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {input_file}")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check that the file format matches the expected structure.")