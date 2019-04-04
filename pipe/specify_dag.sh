#!/bin/bash

set -e

# Clean and tokenise
dvc run -w ..\
  -d data/raw/gtr_projects.csv\
  -d gateway_to_cookies/features/text_preprocessing.py\
  -d gateway_to_cookies/data/make_dataset.py\
  -o data/processed/gtr_tokenised.csv\
  python gateway_to_cookies/data/make_dataset.py -n 10

# Train word embeddings
dvc run -w .. \
  -d data/processed/gtr_tokenised.csv\
  -d gateway_to_cookies/features/w2v.py\
  -d gateway_to_cookies/features/build_features.py\
  -o models/gtr_w2v\
  -o data/processed/gtr_embedding.csv\
  python gateway_to_cookies/features/build_features.py

# Test-train split
dvc run -w ..\
  -d data/processed/gtr_embedding.csv\
  -d data/processed/gtr_tokenised.csv\
  -d gateway_to_cookies/models/train_test_split.py\
  -o data/processed/gtr_train.csv\
  -o data/processed/gtr_test.csv\
  python gateway_to_cookies/models/train_test_split.py

# Train model
dvc run -w ..\
  -d data/processed/gtr_train.csv\
  -d gateway_to_cookies/models/predict_model.py\
  -o models/gtr_forest.pkl\
  python gateway_to_cookies/models/train_model.py

# Evaluate model
touch ../models/metrics.txt
dvc run -w ..\
  -d data/processed/gtr_test.csv\
  -d models/gtr_forest.pkl\
  -d gateway_to_cookies/models/evaluate.py\
  -M models/metrics.txt\
  -f Dvcfile\
  python gateway_to_cookies/models/evaluate.py
