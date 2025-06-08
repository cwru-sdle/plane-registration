#!/bin/bash

REQ_FILE="${1:-requirements.txt}"
PYTHON_PATH="${2:-$HOME/.pyenv/versions/3.9.14/bin/python}"

# Extract prefix from REQ_FILE (e.g., "clustering" from "clustering-requirements.txt")
REQ_PREFIX=$(basename "$REQ_FILE" | sed 's/-requirements\.txt$//')

# Default WHEELS_DIR based on REQ_PREFIX, or fallback to ./wheels if no match
WHEELS_DIR="${3:-./wheels-${REQ_PREFIX}}"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file '$REQ_FILE' not found."
  exit 1
fi

# Create wheels directory
mkdir -p "$WHEELS_DIR"

echo "Downloading wheels with dependency resolution..."

# First, let's try downloading with full dependency resolution
if "$PYTHON_PATH" -m pip download \
  -r "$REQ_FILE" \
  -d "$WHEELS_DIR" \
  --prefer-binary; then
  echo "Successfully downloaded wheels with dependencies"
else
  echo "Failed to download with dependencies. Trying individual packages..."
  
  # Fallback: try downloading compatible versions
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Extract package name (remove version specifier)
    pkg_name=$(echo "$line" | sed 's/[<>=!].*//')
    
    echo "Attempting to download: $pkg_name"
    if ! "$PYTHON_PATH" -m pip download "$pkg_name" -d "$WHEELS_DIR" --prefer-binary; then
      echo "Warning: Failed to download $pkg_name"
    fi
  done < "$REQ_FILE"
fi

echo "Checking for duplicate packages..."
duplicates=$(find "$WHEELS_DIR" -name "*.whl" -exec basename {} \; | sed 's/-[0-9].*//' | sort | uniq -d)

if [[ -n "$duplicates" ]]; then
  echo "Warning: Found potential conflicts for packages:"
  echo "$duplicates"
  echo ""
  echo "Duplicate wheels found:"
  for pkg in $duplicates; do
    echo "Package: $pkg"
    find "$WHEELS_DIR" -name "${pkg}-*.whl" -exec basename {} \;
    echo ""
  done
else
  echo "No duplicate packages found."
fi

echo "Download completed to $WHEELS_DIR"