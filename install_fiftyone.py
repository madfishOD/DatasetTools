import subprocess
import sys
import os

venv_dir = "venv"
python_exe = os.path.join(venv_dir, "Scripts", "python.exe")

def run(cmd):
    print(f"> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", venv_dir])

    print("Upgrading pip...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing FiftyOne, transformers, and torch...")
    run([python_exe, "-m", "pip", "install", "fiftyone", "transformers", "torch", "hf_xet", "accelerate"])

    print("\n✅ Setup complete. To activate the environment, run:")
    print(f"    {os.path.join(venv_dir, 'Scripts', 'activate.bat')}")

if __name__ == "__main__":
    main()