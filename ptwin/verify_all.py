"""ptwin.verify_all - Verification test suite for all PTwin domains."""
import sys
from ptwin.verify_beam import run_verification as run_beam_verification


def main():
    print("=" * 70)
    print(" PTwin Verification Suite: Idealised Backdoor & PRC Residual Check ")
    print("=" * 70)
    print("\n--- Running Beam Domain Verification ---")
    run_beam_verification()
    print("\n[SUCCESS] Beam verification completed cleanly.")


if __name__ == "__main__":
    main()
