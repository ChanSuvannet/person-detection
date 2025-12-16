#!/bin/bash

###############################################################################
# Person Detection Pipeline
# This script runs the entire project pipeline from data collection to evaluation
###############################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if a step should be run
should_run_step() {
    local step_name=$1
    local marker_file=$2
    
    if [ -f "$marker_file" ]; then
        print_warning "$step_name already completed. Skip? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            return 1  # Skip
        fi
    fi
    return 0  # Run
}

# Function to mark step as complete
mark_complete() {
    local marker_file=$1
    touch "$marker_file"
}

###############################################################################
# Main Pipeline
###############################################################################

print_header "PERSON DETECTION PROJECT - COMPLETE PIPELINE"
echo "This script will guide you through the entire 3-day pipeline."
echo "Total estimated time: 8-16 hours (depending on hardware)"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

###############################################################################
# DATA COLLECTION AND PREPROCESSING
###############################################################################

print_header "DATA COLLECTION AND PREPROCESSING"

# Step 1: Data Collection
if should_run_step "Data Collection" "data/.collection_complete"; then
    print_info "Step 1/7: Downloading Person class images from Open Images..."
    print_info "Expected time: 30-60 minutes"
    print_info "Expected output: ~1500 images"
    
    python src/data_collection.py
    
    if [ $? -eq 0 ]; then
        mark_complete "data/.collection_complete"
        print_success "Data collection complete!"
    else
        print_error "Data collection failed!"
        exit 1
    fi
else
    print_info "Skipping data collection (already complete)"
fi

# Step 2: Data Preprocessing
if should_run_step "Data Preprocessing" "data/.preprocessing_complete"; then
    print_info "Step 2/7: Preprocessing data (duplicate removal, validation, splitting)..."
    print_info "Expected time: 10-20 minutes"
    
    python src/data_preprocessing.py
    
    if [ $? -eq 0 ]; then
        mark_complete "data/.preprocessing_complete"
        print_success "Data preprocessing complete!"
    else
        print_error "Data preprocessing failed!"
        exit 1
    fi
else
    print_info "Skipping data preprocessing (already complete)"
fi

print_success "DAY 1 COMPLETE!"
print_info "You now have clean, organized data ready for training."
print_info "Data location: data/processed/"

###############################################################################
# QUALITY ENHANCEMENT AND TRAINING
###############################################################################

print_header "QUALITY ENHANCEMENT AND TRAINING"

# Step 3: Data Quality Enhancement
if should_run_step "Quality Enhancement" "data/.enhancement_complete"; then
    print_info "Step 3/7: Enhancing image quality..."
    print_info "Expected time: 15-30 minutes"
    print_info "Enhancements: brightness, contrast, sharpness"
    
    python src/data_quality.py
    
    if [ $? -eq 0 ]; then
        mark_complete "data/.enhancement_complete"
        print_success "Quality enhancement complete!"
    else
        print_error "Quality enhancement failed!"
        exit 1
    fi
else
    print_info "Skipping quality enhancement (already complete)"
fi

# Step 4: Train on Original Dataset
if should_run_step "Training (Original)" "models/.train_original_complete"; then
    print_warning "Step 4/7: Training model on ORIGINAL dataset..."
    print_info "Expected time: 2-4 hours (GPU) or 8-12 hours (CPU)"
    print_info "This will take a while. You can monitor progress in another terminal:"
    print_info "  docker-compose run app tail -f models/original/train/results.csv"
    echo ""
    print_warning "Continue? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python src/train.py --data original
        
        if [ $? -eq 0 ]; then
            mark_complete "models/.train_original_complete"
            print_success "Training on original data complete!"
        else
            print_error "Training on original data failed!"
            exit 1
        fi
    else
        print_warning "Skipping original training. You can run it later with:"
        print_info "  docker-compose run app python src/train.py --data original"
    fi
else
    print_info "Skipping training on original data (already complete)"
fi

# Step 5: Train on Enhanced Dataset
if should_run_step "Training (Enhanced)" "models/.train_enhanced_complete"; then
    print_warning "Step 5/7: Training model on ENHANCED dataset..."
    print_info "Expected time: 2-4 hours (GPU) or 8-12 hours (CPU)"
    echo ""
    print_warning "Continue? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python src/train.py --data enhanced
        
        if [ $? -eq 0 ]; then
            mark_complete "models/.train_enhanced_complete"
            print_success "Training on enhanced data complete!"
        else
            print_error "Training on enhanced data failed!"
            exit 1
        fi
    else
        print_warning "Skipping enhanced training. You can run it later with:"
        print_info "  docker-compose run app python src/train.py --data enhanced"
    fi
else
    print_info "Skipping training on enhanced data (already complete)"
fi

print_success "DAY 2 COMPLETE!"
print_info "Both models have been trained. Time for evaluation!"

###############################################################################
# EVALUATION AND COMPARISON
###############################################################################

print_header "EVALUATION AND COMPARISON"

# Step 6: Model Evaluation
if should_run_step "Evaluation" "results/.evaluation_complete"; then
    print_info "Step 6/7: Evaluating models and creating comparison..."
    print_info "Expected time: 20-40 minutes"
    
    python src/evaluate.py
    
    if [ $? -eq 0 ]; then
        mark_complete "results/.evaluation_complete"
        print_success "Evaluation complete!"
    else
        print_error "Evaluation failed!"
        exit 1
    fi
else
    print_info "Skipping evaluation (already complete)"
fi

# Step 7: Generate Final Report
print_info "Step 7/7: Generating final summary..."

cat << 'EOF'

================================================================================
                        PIPELINE EXECUTION COMPLETE!
================================================================================

📊 RESULTS SUMMARY:

Data Processing:
  ✓ Images collected and preprocessed
  ✓ Quality enhancements applied
  ✓ Train/Val/Test splits created

Model Training:
  ✓ Model trained on original dataset
  ✓ Model trained on enhanced dataset

Evaluation:
  ✓ Performance metrics calculated
  ✓ Visualizations generated
  ✓ Comparison analysis complete

📁 KEY OUTPUT LOCATIONS:

  📦 Data:
     - Original: data/processed/
     - Enhanced: data/enhanced/

  🤖 Models:
     - Original: models/original/best.pt
     - Enhanced: models/enhanced/best.pt

  📈 Results:
     - Metrics: results/metrics/
     - Visualizations: results/visualizations/
     - Comparison: results/comparison/

🎯 NEXT STEPS:

  1. Review comparison results:
     $ cat results/comparison/improvement_summary.txt

  2. View detection visualizations:
     $ ls results/visualizations/

  3. Prepare presentation using:
     - Comparison plots in results/comparison/
     - Detection samples in results/visualizations/
     - Metrics from results/metrics/

  4. Run inference on new images:
     $ docker-compose run app python -c "
     from ultralytics import YOLO
     model = YOLO('models/enhanced/best.pt')
     results = model.predict('your_image.jpg')
     "

📊 VIEW RESULTS IN BROWSER:

  Copy the results folder to your local machine to view images and plots:
  
  From your host machine:
  $ docker cp person-detection:/app/results ./results_output

  Then open the PNG files in results_output/comparison/ and 
  results_output/visualizations/

================================================================================
                    ✨ PROJECT COMPLETE! GREAT WORK! ✨
================================================================================

EOF

print_success "All steps completed successfully!"
print_info "Check the locations above for your results."

# Create completion marker
mark_complete ".pipeline_complete"