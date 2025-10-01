import shutil
import subprocess
import sys
import os

venv_dir = "venv"
python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
plugins_repo = "fiftyone-plugins"
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

    if not os.path.exists(plugins_repo):
        print(f"Cloning {plugins_repo} repo...")
        run(["git", "clone", "https://github.com/voxel51/fiftyone-plugins", plugins_repo])
    else:
        print(f"Using existing clone at '{plugins_repo}'")

    plugins_root = os.path.join(os.getcwd(), plugins_repo, "plugins")
    fo_plugin_dir = os.path.join(os.getcwd(), venv_dir, "Lib", "site-packages", "fiftyone", "plugins", "local")

    os.makedirs(fo_plugin_dir, exist_ok=True)

    for plugin_name in core_plugins:
        src = os.path.join(plugins_root, plugin_name)
        dst = os.path.join(fo_plugin_dir, plugin_name)
        print(f"> Installing plugin: {plugin_name}")
        print(f"  Source: {src}")
        print(f"  Destination: {dst}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if os.path.exists(src):
            shutil.copytree(src, dst)
        else:
            print(f"❌ Plugin path does not exist: {src}")

def main():
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", venv_dir])

    print("Upgrading pip...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

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