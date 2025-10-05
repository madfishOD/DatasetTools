import fiftyone as fo
import fiftyone.plugins as fopl
import fiftyone.plugins.utils as fopu


plugin_name = "hello_python"

def setup_my_plugins() -> bool:
    try:
        fopl.download_plugin(
                "https://github.com/madfishOD/FiftyOnePlugins.git",
                plugin_names= "@madfish/hello_python",
                overwrite=True
            )
        fopl.enable_plugin("@madfish/hello_python")
    except Exception as e:
        print(f"❌ Failed to install plugin @madfish/hello_python: {e}")

def main():
    fopl.enable_plugin("@madfish/mad-io")
    plugins_list = fopl.list_enabled_plugins()
    print(plugins_list)

    session = fo.launch_app()
    session.wait()

if __name__ == "__main__":
    main()
