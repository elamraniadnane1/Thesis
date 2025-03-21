import pandas as pd

def xlsx_to_csv(input_file_path, output_file_path):
    # Read the Excel file
    df = pd.read_excel(input_file_path)
    
    # Write the data to a CSV file
    df.to_csv(output_file_path, index=False)
    
# Example usage:
if __name__ == "__main__":
    input_file = "REMACTO Projects.xlsx"
    output_file = "REMACTO Projects.csv"
    xlsx_to_csv(input_file, output_file)
    print(f"Conversion complete. CSV saved at {output_file}.")
