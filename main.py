#!/usr/bin/env python3
"""
Face Detection & Recognition — One-Stop Entry Point

Run the full pipeline: enroll faces → recognize faces.
"""
import sys


def print_banner():
    print()
    print("=" * 50)
    print("  FACE DETECTION & RECOGNITION SYSTEM")
    print("=" * 50)
    print()


def print_menu():
    print("  [1] Enroll faces (collect images & train model)")
    print("  [2] Face recognition")
    print("  [3] Exit")
    print()


def main():
    print_banner()
    print_menu()

    while True:
        try:
            choice = input("Select option (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if choice == "1":
            from src.collect import run as collect_run
            from src.train import run as train_run
            collect_run()
            train_run()
        elif choice == "2":
            from src.recognize import run
            run()
        elif choice == "3":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option. Enter 1, 2, or 3.")

        print()
        print_menu()


if __name__ == "__main__":
    main()
