import argparse
import json
import uvicorn

from inference_node.agentic_pipeline import AgenticPlannerPipeline
from inference_node.api import create_app
from inference_node.config import InferenceConfig
from inference_node.frame_extractor import FrameExtractor
from inference_node.retrieval.encoder import get_retrieval_encoder
from inference_node.retrieval.search import RetrievalEngine
from inference_node.retrieval.vector_store import VectorStore
from inference_node.vqa import get_vqa_reasoner
from shared.utils import setup_logger


def run_inference_node(config: InferenceConfig) -> None:
    logger = setup_logger("InferenceNode")
    logger.info("Starting Agentic Planning VLM Inference Node")
    logger.info(f"Retrieval model: {config.retrieval_model}")
    logger.info(f"Reasoning model: {config.reasoning_model}")
    logger.info(f"PostgreSQL pgvector Table: {config.postgres_table}")
    logger.info(f"Video sources: {config.video_sources}")

    logger.info("Connecting to PostgreSQL VectorStore...")
    vector_store = VectorStore(
        table_name=config.postgres_table,
        postgres_url=config.postgres_url,
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
    )

    logger.info(f"Loading retrieval encoder '{config.retrieval_model}'...")
    encoder = get_retrieval_encoder(
        model_name=config.retrieval_model,
        device=config.device,
    )

    retrieval_engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=config.metadata_filter_enabled,
    )

    logger.info("Initializing frame extractor...")
    frame_extractor = FrameExtractor(video_sources=config.video_sources)

    logger.info(f"Loading Agentic VLM reasoner '{config.reasoning_model}'...")
    reasoner = get_vqa_reasoner(
        model_name=config.reasoning_model,
        device=config.device,
    )

    logger.info("Building Agentic Planner Pipeline...")
    pipeline = AgenticPlannerPipeline(
        retrieval_engine=retrieval_engine,
        vector_store=vector_store,
        frame_extractor=frame_extractor,
        reasoner=reasoner,
        max_planning_steps=config.max_planning_steps,
    )

    logger.info("Starting Agentic API server...")
    app = create_app(pipeline=pipeline, vector_store=vector_store)
    uvicorn.run(app, host="0.0.0.0", port=config.api_port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV Agentic Planning VLM Inference Node")
    parser.add_argument("--postgres_table", type=str, default="track_events")
    parser.add_argument("--postgres_url", type=str, default=None)
    parser.add_argument("--supabase_url", type=str, default=None)
    parser.add_argument("--supabase_key", type=str, default=None)
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-base-patch16-224",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="openai-5.6",
        help="Options: 'openai-5.6', 'gemini-2.5-flash', 'Qwen/Qwen3-VL-8B-Instruct'",
    )
    parser.add_argument("--api_port", type=int, default=8100)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--max_planning_steps", type=int, default=5)
    parser.add_argument(
        "--metadata_filter_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--video_sources",
        type=str,
        default="{}",
        help='JSON dict mapping camera_id to video path, e.g. \'{"cam_1": "/app/dataset/video.avi"}\'',
    )
    args = parser.parse_args()

    config = InferenceConfig(
        postgres_table=args.postgres_table,
        postgres_url=args.postgres_url,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        retrieval_model=args.retrieval_model,
        reasoning_model=args.reasoning_model,
        video_sources=json.loads(args.video_sources),
        retrieval_top_k=args.retrieval_top_k,
        max_planning_steps=args.max_planning_steps,
        metadata_filter_enabled=args.metadata_filter_enabled,
        api_port=args.api_port,
        device=args.device,
    )
    run_inference_node(config)
