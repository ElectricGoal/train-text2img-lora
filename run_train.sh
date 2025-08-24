#!/bin/bash

# Simple script to run LoRA training for Stable Diffusion
# Usage: ./run_train.sh

# Set default values
PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"
TRAIN_DATA_DIR="./data"  # Change this to your dataset path
OUTPUT_DIR="./output"
RESOLUTION=512
BATCH_SIZE=4
NUM_EPOCHS=50
LEARNING_RATE=1e-4
RANK=16
VALIDATION_PROMPT="black wavy french bob vibes from 1920s"

# Check if dataset directory exists
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "Error: Dataset directory '$TRAIN_DATA_DIR' not found!"
    echo "Please create a dataset directory with the following structure:"
    echo "dataset/"
    echo "├── image1.jpg"
    echo "├── image2.jpg"
    echo "└── metadata.jsonl"
    echo ""
    echo "The metadata.jsonl should contain:"
    echo '{"file_name": "image1.jpg", "text": "description of image1"}'
    echo '{"file_name": "image2.jpg", "text": "description of image2"}'
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting LoRA training..."
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Training data: $TRAIN_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Resolution: $RESOLUTION"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA rank: $RANK"
echo ""

# Run the training script
python train_text2img_lora.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --rank $RANK \
    --validation_prompt "$VALIDATION_PROMPT" \
    --validation_epochs 50 \
    --checkpointing_steps 400 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --enable_xformers_memory_efficient_attention \
    --report_to wandb

echo "Training completed! Check the output in: $OUTPUT_DIR"
