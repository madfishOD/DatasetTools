import fiftyone as fo
import fiftyone.plugins as fopl
import os

plugins_dir = os.path.abspath("venv/fiftyone_plugins")
os.makedirs(plugins_dir, exist_ok=True)
fo.config.plugins_dir = plugins_dir

core_plugins = [
    "annotation", "brain", "dashboard", "evaluation", "io",
    "indexes", "delegated", "runs", "utils", "zoo"
]

print("================================")
print(" Installing FiftyOne Core Plugins")
print("================================")

for plugin_name in core_plugins:
    print(f"> Installing plugin: {plugin_name}")
    try:
        fopl.download_plugin(
            "https://github.com/voxel51/fiftyone-plugins",
            plugin_names=[f"@voxel51/{plugin_name}"]
        )
    except Exception as e:
        print(f"❌ Failed to install plugin {plugin_name}: {e}")