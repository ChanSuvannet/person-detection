#!/bin/bash
set -e

echo "🚀 Starting Person Detection Pipeline"

# Helper
run_step () {
  NAME=$1
  MARKER=$2
  CMD=$3

  if [ -f "$MARKER" ]; then
    echo "⏭️  Skipping $NAME (already completed)"
    return
  fi

  echo "▶️  Running $NAME"
  eval "$CMD"
  touch "$MARKER"
  echo "✅ $NAME completed"
}

# ------------------------
# DATA PIPELINE
# ------------------------
run_step "Data Collection" \
  data/.collection_complete \
  "python src/data_collection.py"

run_step "Data Preprocessing" \
  data/.preprocessing_complete \
  "python src/data_preprocessing.py"

run_step "Quality Enhancement" \
  data/.enhancement_complete \
  "python src/data_quality.py"

# ------------------------
# TRAINING
# ------------------------
run_step "Training (Original)" \
  models/.train_original_complete \
  "python src/train.py --data original"

run_step "Training (Enhanced)" \
  models/.train_enhanced_complete \
  "python src/train.py --data enhanced"

# ------------------------
# EVALUATION
# ------------------------
run_step "Evaluation" \
  results/.evaluation_complete \
  "python src/evaluate.py"

touch .pipeline_complete

echo "🎉 PIPELINE FINISHED SUCCESSFULLY"
