import fiftyone as fo

def main():
    session = fo.launch_app()
    session.wait()

if __name__ == "__main__":
    main()
