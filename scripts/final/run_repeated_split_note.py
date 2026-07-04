from __future__ import annotations


def main() -> int:
    print("The full repeated-split CNN robustness run was executed in Colab.")
    print("Local smoke-test commands are provided.")
    print()
    print("Colab notebook:")
    print("  notebooks/colab_repeated_split_robustness.ipynb")
    print()
    print("Full project runner command:")
    print(
        "  python experiments/79_run_repeated_split_robustness_colab.py "
        "--seeds 42 101 202 --densities 0.01 0.03 0.05 0.10 "
        "--top-k 32 --epochs 8 --patience 3 --max-train-wafers 2500 "
        "--max-test-wafers 500 --batch-size 64 --hidden-channels 32 "
        "--point-estimators 100 --use-amp"
    )
    print()
    print("Local lightweight RF demo:")
    print("  python scripts/final/run_final_demo.py --max-test-wafers 100")
    print()
    print("Local CNN smoke test:")
    print("  python scripts/final/run_cnn_smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
