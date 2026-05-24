import os
import sys

def detect_environment():
    """
    Detects if the code is running inside Google Colab or locally.
    Returns:
        tuple: (env_name, data_dir)
    """
    try:
        import google.colab
        # Running in Google Colab
        env_name = "colab"
        # Standard Drive mount path
        drive_path = '/content/gdrive/MyDrive/Time series Projet/data'
        if not os.path.exists(drive_path):
            # Mount drive if not already mounted
            try:
                from google.colab import drive
                drive.mount('/content/gdrive')
            except Exception as e:
                print(f"Warning: Failed to mount Google Drive: {e}")
        data_dir = drive_path
    except ImportError:
        # Running locally or elsewhere
        env_name = "local"
        # Check standard local paths relative to workspace root
        # The workspace root is 'C:\Users\admin\Downloads\Time series Projet'
        # We can look for the 'data' directory in the current working directory or parent directories
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        if not os.path.exists(data_dir):
            # Fallback to current working directory 'data'
            data_dir = os.path.abspath("./data")
            
    return env_name, data_dir

# Global configuration
ENVIRONMENT, DATA_DIR = detect_environment()
print(f"Environment detected: {ENVIRONMENT}")
print(f"Data directory path: {DATA_DIR}")

# Figures directory configuration
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures directory path: {FIGURES_DIR}")

# Key parameters
# Primary prediction target : monthly tourist arrivals
TARGET_COL = 'Arrivals'
# Secondary prediction target : monthly overnight stays (nuitées)
NIGHTS_COL  = 'Nights'
START_DATE = '1995-01-01'
END_DATE = '2026-04-01'
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
