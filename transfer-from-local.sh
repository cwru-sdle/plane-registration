#!/bin/bash
# batch-split.sh
mkdir -p batches
cd /mnt/external/aconity/AC3D\ Camera

MAX=100000000000  # ~100 GB in bytes
acc=0
i=0

mkdir -p ../batches/batch_$i

for f in *; do
    size=$(stat -c%s "$f")
    if (( acc + size > MAX )); then
        ((i++))
        mkdir -p ../batches/batch_$i
        acc=0
    fi
    mv "$f" ../batches/batch_$i/
    ((acc += size))
done

for d in /mnt/external/aconity/batches/*; do
  tar -cf - -C "$d" . | ssh -J aml334@pioneer aml334@cradle3202t \
    "hdfs dfs -put - /user/aml334/$(basename $d).tar"
done
