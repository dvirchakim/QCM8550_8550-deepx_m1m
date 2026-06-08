import zipfile
import os

def package():
    with zipfile.ZipFile('deploy.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Files to include explicitly or patterns
        include_dirs = ['src', 'assets']
        include_files = ['config.py', 'main.py', 'requirements.txt', 'README.md']

        for file in include_files:
            if os.path.exists(file):
                zipf.write(file, file)
        
        for d in include_dirs:
            for root, dirs, files in os.walk(d):
                if '__pycache__' in root:
                    continue
                for file in files:
                    if file.endswith('.pyc'):
                        continue
                    file_path = os.path.join(root, file)
                    # Force forward slashes for Linux compatibility
                    arcname = file_path.replace(os.sep, '/')
                    zipf.write(file_path, arcname)
    print("Created deploy.zip with POSIX paths.")

if __name__ == '__main__':
    package()
