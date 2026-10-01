import subprocess
import os
import sys

def submit_to_kaggle(file_path='submission_ensemble.csv', message='Top GBDT Ensemble with Domain Feature Engineering'):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist!")
        return
        
    print(f"Submitting {file_path} to Kaggle...")
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_890ba420602fcb8048bfe87686d1ed04'
    
    cmd = f'kaggle competitions submit -c playground-series-s6e9 -f "{file_path}" -m "{message}"'
    res = subprocess.run(cmd, shell=True)
    
    print("\nFetching latest submissions status...")
    status_cmd = 'kaggle competitions submissions -c playground-series-s6e9'
    subprocess.run(status_cmd, shell=True)


if __name__ == '__main__':
    file_to_sub = sys.argv[1] if len(sys.argv) > 1 else 'submission_ensemble.csv'
    msg = sys.argv[2] if len(sys.argv) > 2 else 'GBDT Ensemble'
    submit_to_kaggle(file_to_sub, msg)
