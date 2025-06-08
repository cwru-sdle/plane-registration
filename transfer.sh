#!/bin/bash
# batch-split.sh - Splits files into batches by size and transfers to HDFS via SSH jump
set -e  # Exit on any error

# ====== CONFIGURATION ======
REMOTE_USER="aml334"                 # Your remote username
JUMP_HOST="pioneer.case.edu"        # SSH jump host
REMOTE_HOST="cradle3202t.priv.cwru.edu"  # Final remote host

SOURCE_DIR="/mnt/external/aconity/camera"  # Local directory with source files
BATCH_DIR="/mnt/external/aconity/batches"  # Local directory to store batches temporarily
MAX=$((100 * 1024 * 1024 * 1024))            # Max batch size in bytes (100 GB)

KEYTAB_PATH="/home/aml334/aml334.keytab"     # Path to your keytab file on remote host
KERBEROS_PRINCIPAL="aml334@ADS.CASE.EDU"     # Your Kerberos principal
HDFS_USER="aml334"                           # HDFS user, usually same as REMOTE_USER

# ====== PRECHECKS ======
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

if ! ls "$SOURCE_DIR"/* >/dev/null 2>&1; then
    echo "Error: No files found in '$SOURCE_DIR'"
    exit 1
fi

echo "Creating batches from files in: $SOURCE_DIR"

mkdir -p "$BATCH_DIR"
cd "$SOURCE_DIR"

# ====== SPLIT FILES INTO BATCHES ======
acc=0
batch_index=0
mkdir -p "$BATCH_DIR/batch_$batch_index"

for file in *; do
    [[ -f "$file" ]] || continue
    
    size=$(stat -c%s "$file")
    echo "Processing: $file (${size} bytes)"
    
    if (( acc + size > MAX )); then
        echo "Batch $batch_index full (${acc} bytes), starting batch $((batch_index + 1))"
        ((batch_index++))
        mkdir -p "$BATCH_DIR/batch_$batch_index"
        acc=0
    fi
    
    mv "$file" "$BATCH_DIR/batch_$batch_index/"
    ((acc += size))
done

echo "Created $((batch_index + 1)) batches"

# ====== TRANSFER TO HDFS VIA SSH JUMP ======
echo "Transferring batches to HDFS..."

for batch_path in "$BATCH_DIR"/*; do
    [[ -d "$batch_path" ]] || continue
    batch_name=$(basename "$batch_path")
    
    if ! ls "$batch_path"/* >/dev/null 2>&1; then
        echo "Warning: $batch_name is empty, skipping"
        continue
    fi
    
    echo "Transferring $batch_name..."

    # Debug: print SSH command (optional)
    echo "Debug: ssh -J $REMOTE_USER@$JUMP_HOST $REMOTE_USER@$REMOTE_HOST"

    if tar -cf - -C "$batch_path" . | \
       ssh -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$REMOTE_HOST" bash -c "'
         kinit -kt \"$KEYTAB_PATH\" \"$KERBEROS_PRINCIPAL\" && \
         hdfs dfs -put - /user/$HDFS_USER/${batch_name}.tar
       '"; then
        echo "Successfully transferred $batch_name"
    else
        echo "Error transferring $batch_name"
    fi
done

echo "Transfer complete!"
