#!/bin/bash
# batch-split.sh - Improved version with error handling

set -e  # Exit on any error

SOURCE_DIR="/mnt/external/aconity/camera"
BATCH_DIR="/mnt/external/aconity/batches"
MAX=100000000000  # ~100 GB in bytes

# Check if source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Check if there are files to process
if ! ls "$SOURCE_DIR"/* >/dev/null 2>&1; then
    echo "Error: No files found in '$SOURCE_DIR'"
    exit 1
fi

echo "Creating batches from files in: $SOURCE_DIR"

# Create batch directory
mkdir -p "$BATCH_DIR"
cd "$SOURCE_DIR"

acc=0
i=0
mkdir -p "$BATCH_DIR/batch_$i"

echo "Processing files..."
for f in *; do
    # Skip if not a regular file
    [[ -f "$f" ]] || continue
    
    size=$(stat -c%s "$f")
    echo "Processing: $f (${size} bytes)"
    
    if (( acc + size > MAX )); then
        echo "Batch $i full (${acc} bytes), starting batch $((i+1))"
        ((i++))
        mkdir -p "$BATCH_DIR/batch_$i"
        acc=0
    fi
    
    mv "$f" "$BATCH_DIR/batch_$i/"
    ((acc += size))
done

echo "Created $((i+1)) batches"

# Transfer batches to HDFS
echo "Transferring batches to HDFS..."
for d in "$BATCH_DIR"/*; do
    if [[ -d "$d" ]]; then
        batch_name=$(basename "$d")
        echo "Transferring $batch_name..."
        
        # Check if batch has files
        if ! ls "$d"/* >/dev/null 2>&1; then
            echo "Warning: $batch_name is empty, skipping"
            continue
        fi
        
        # Create tar and transfer via SSH
        if tar -cf - -C "$d" . | ssh -J aml334@pioneer aml334@cradle3202t \
            "hdfs dfs -put - /user/aml334/${batch_name}.tar"; then
            echo "Successfully transferred $batch_name"
        else
            echo "Error transferring $batch_name"
        fi
    fi
done

echo "Transfer complete!"
