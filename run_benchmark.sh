#!/usr/bin/env bash

mkdir -p artifacts

rm -rf artifacts/crops
rm -f artifacts/registry.*
rm -f artifacts/cctv_vlm.db


# stage 1: ingest video feed produces -> reid embeddings + compressed tracks + other metadata
poetry run python scripts/run_reid_pipeline.py \
--reid_model_type vitclipreid \
--dir dataset/S02 \
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
--video-dir dataset/S02 \
--output-dir artifacts/crops

# stage 4: takes the crops, runs visual encoder and produces retrieval embeddings + final identities 
poetry run python scripts/produce_retrieval_embeddings.py \
--crop_dir artifacts/crops \
--retrieval_model openai/clip-vit-large-patch14 \
--global_match_json artifacts/registry.tracks.global_matches.json \
--output_npz artifacts/registry.retrieval.embeddings.npz \
--output_json artifacts/registry.tracks.identities.json \
--target_k 3

# stage 5: compile everything into a sqlite database with S02 ground truths
poetry run python scripts/build_cctv_database.py \
--artifacts_dir artifacts \
--output_db artifacts/cctv_vlm.db \
--test_tracks_json dataset/tracks.json

# stage 6: run NL retrieval benchmark against the compiled database ground truths
poetry run python scripts/benchmark_nl_retrieval.py \
--queries_file dataset/tracks.json \
--db_path artifacts/cctv_vlm.db \
--top_k 10