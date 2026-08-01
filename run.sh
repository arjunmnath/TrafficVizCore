#!/usr/bin/env bash

mkdir -p artifacts

rm -rf artifacts/crops
rm -f artifacts/registry.*


# stage 1: ingest video feed produces -> reid embeddings + compressed tracks + other metadata
poetry run python scripts/run_reid_pipeline.py \
--reid_model_type vitclipreid \
--dir input_vids \
--output artifacts/registry.tracks.models.json \
--output_npz artifacts/registry.reid.embeddings.npz \
--device mps \
--fp16 \
--tracker bytetrackx.yaml \
--intra-camera-threshold 0.6

# stage 2: takes the reid embeddings and produces global matches across all cameras
poetry run python scripts/match_multicamera.py \
--json artifacts/registry.tracks.models.json \
--npz artifacts/registry.reid.embeddings.npz \
--threshold 0.6 \
--output artifacts/registry.tracks.global_matches.json

# stage 3: takes the compressed track details and produced the crops of each tracks
poetry run python scripts/crop_tracks.py \
--registry artifacts/registry.tracks.models.json \
--video-dir input_vids \
--output-dir artifacts/crops

# stage 4: takes the crops, runs visual encoder and produces retrieval embeddings + final identities 
poetry run python scripts/produce_retrieval_embeddings.py \
--crop_dir artifacts/crops \
--retrieval_model openai/clip-vit-large-patch14 \
--global_match_json artifacts/registry.tracks.global_matches.json \
--output_npz artifacts/registry.retrieval.embeddings.npz \
--output_json artifacts/registry.tracks.identities.json \
--target_k 3