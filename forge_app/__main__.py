import argparse

from .app import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch OpenVoice Forge")
    parser.add_argument("--share", action="store_true", help="Enable a temporary Gradio share URL")
    args = parser.parse_args()
    launch(share=args.share)


if __name__ == "__main__":
    main()
