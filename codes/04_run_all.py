import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

scripts = [
    "01_data_cleaning_eda.py",
    "02_model_training.py",
    "03_business_threshold_segments.py"
]

for script in scripts:
    script_path = PROJECT_DIR / script

    print("\n" + "=" * 70)
    print(f"Running {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:
        print(f"\nError while running {script}")
        sys.exit(result.returncode)

print("\nAll scripts completed successfully.")