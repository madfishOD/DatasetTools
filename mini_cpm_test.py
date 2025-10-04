import fiftyone as fo
import fiftyone.zoo as foz

def register_model():
    print("Running register_model()")
    # Register the MiniCPM-V model source
    foz.register_zoo_model_source(
        "https://github.com/harpreetsahota204/minicpm-v", 
        overwrite=True
    )

def load_model():
    print("Running load_model()")
    # Load the model (downloads on first use)
    model = foz.load_zoo_model(
        "openbmb/MiniCPM-V-4_5",
        install_requirements=True 
    )

def is_model_registered() -> bool:
    return "openbmb/MiniCPM-V-4_5" in foz.list_zoo_models()

def main():
    print("Running main()")
    registered = is_model_registered()
    print(f"MiniCPM-V registered is {registered}!")

    if(not registered):
        register_model()

if __name__ == "__main__":
    main()