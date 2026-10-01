#!/usr/bin/env bash
set -e

mkdir -p data/raw
kaggle datasets download -d laotse/credit-risk-dataset -p data/raw --unzip
echo "Done: data/raw/credit_risk_dataset.csv"