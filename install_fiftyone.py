import subprocess
import sys
import os

venv_dir = "venv"
python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
plugins_repo = os.path.join(venv_dir, "fiftyone-plugins")
core_plugins = [
    "annotation", "brain", "dashboard", "evaluation", "io",
    "indexes", "delegated", "runs", "utils", "zoo"
]
dependencies_file = "fo_dependencies.txt"


def run(cmd):
    print(f"> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def read_dependencies(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    
def install_core_plugins():
    print("================================")
    print(" Installing FiftyOne Plugins")
    print("================================")

    plugin_install_script = """
import fiftyone.plugins as fopl
core_plugins = [
    "annotation", "brain", "dashboard", "evaluation", "io",
    "indexes", "delegated", "runs", "utils", "zoo"
]
for plugin_name in core_plugins:
    print(f"> Installing plugin: {plugin_name}")
    try:
        fopl.download_plugin(
            "https://github.com/voxel51/fiftyone-plugins",
            plugin_names=[f"@voxel51/{plugin_name}"]
        )
    except Exception as e:
        print(f"❌ Failed to install plugin {plugin_name}: {e}")
"""

    run([python_exe, "-c", plugin_install_script])
    

def main():
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", venv_dir])

    print("Upgrading pip...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "build"])

    print("Installing FiftyOne...")
    run([python_exe, "-m", "pip", "install", "fiftyone"])

    print("Upgrading FiftyOne...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "fiftyone"])

    print(f"Reading dependencies from '{dependencies_file}'...")
    deps = read_dependencies("fo_dependencies.txt")
    print(f"Installing: {deps}")
    run([python_exe, "-m", "pip", "install"] + deps)

    install_core_plugins()

    print("\n✅ Setup complete. To activate the environment, run:")
    print(f"    {os.path.join(venv_dir, 'Scripts', 'activate.bat')}")

if __name__ == "__main__":
    main()