import subprocess

# Path to your batch file
batch_file = r"C:\Users\DELL\OneDrive\Desktop\Thesis\create_python_files_civiccatalyst.bat"

# Execute the batch file
result = subprocess.run(batch_file, shell=True, capture_output=True, text=True)

# Print the output from the batch file
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
