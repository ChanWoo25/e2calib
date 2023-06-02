#! /bin/bash

# for INDEX in 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029; do
#   python3 offline_reconstruction.py \
#     --h5file /data/datasets/dataset_celepixel/230601_calib_01/230601_calib_01_${INDEX}.h5 \
#     --freq_hz 5 --upsample_rate 4 \
#     --height 800 --width 1280
# done

for INDEX in 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029; do
  python3 offline_reconstruction.py \
    --h5file /data/datasets/dataset_celepixel/230601_calib_02/230601_calib_02_${INDEX}.h5 \
    --freq_hz 5 --upsample_rate 4 \
    --height 800 --width 1280
done

