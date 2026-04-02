"""
main.py — Entry Point
House Price Prediction | Vimal Sahani
Usage:
    python main.py --mode train
    python main.py --mode predict --input data/test.csv
"""

import argparse
from src.preprocess import build_pipeline
from src.train import train
from src.predict import predict


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument("--mode",  choices=["train", "predict"], required=True)
    parser.add_argument("--train", default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--test",  default="data/test.csv",  help="Path to test CSV")
    parser.add_argument("--output",default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    print("=" * 50)
    print("  🏠 House Price Prediction — Vimal Sahani")
    print("=" * 50)

    X_train, y_train, X_test, test_ids = build_pipeline(args.train, args.test)

    if args.mode == "train":
        train(X_train, y_train)

    elif args.mode == "predict":
        predict(X_test, test_ids, output_path=args.output)


if __name__ == "__main__":
    main()
