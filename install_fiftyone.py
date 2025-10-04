import subprocess
import sys
import os

venv_dir = "venv"
python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
plugins_dir = os.path.join(venv_dir, "fiftyone_plugins")
dependencies_file = "fo_dependencies.txt"


def run(cmd):
    print(f"> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def read_dependencies(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    

def main():
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", venv_dir])

    print("Upgrading pip...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "build"])

    print("Installing FiftyOne...")
    run([python_exe, "-m", "pip", "install", "fiftyone"])

    print("Upgrading FiftyOne...")
    run([python_exe, "-m", "pip", "install", "-U", "fiftyone"])

    print(f"Reading dependencies from '{dependencies_file}'...")
    deps = read_dependencies("fo_dependencies.txt")
    print(f"Installing: {deps}")
    run([python_exe, "-m", "pip", "install"] + deps)

    print("\n✅ Setup complete. To activate the environment, run:")
    print(f"    {os.path.join(venv_dir, 'Scripts', 'activate.bat')}")

if __name__ == "__main__":
    main()