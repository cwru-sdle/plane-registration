#!/bin/bash

REQ_FILE="${1:-requirements.txt}"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file '$REQ_FILE' not found."
  exit 1
fi

while read -r pkg || [[ -n "$pkg" ]]; do
  ~/.pyenv/versions/3.9.14/bin/python -m pip download "$pkg" -d ./wheels-3.9 || echo "Skipping $pkg"
done < "$REQ_FILE"
