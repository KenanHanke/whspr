#!/usr/bin/env python3
from . import client
from . import server
import argparse


def main():
    parser = argparse.ArgumentParser(description="A dictation software package for Linux. Running this command without arguments either starts a new recording or ends one, copying the result to the clipboard.")
    parser.add_argument("--paste", action="store_true", help="run as normal but also automatically paste the transcription result in the currently focussed application")
    parser.add_argument("--cancel", action="store_true", help="cancel the ongoing recording")
    parser.add_argument("--finish-setup", action="store_true", help=argparse.SUPPRESS)  # hidden argument
    args = parser.parse_args()

    if args.finish_setup:
        server.load_model()  # load the model to finish setup and then exit
        return
    
    # try to start the server; this will fail silently if it's already up
    server.start()

    if args.cancel:
        client.cancel_recording()
    else:
        client.main(paste=args.paste)


if __name__ == "__main__":
    main()
