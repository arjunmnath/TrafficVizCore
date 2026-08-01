import argparse
import json

from shared.utils import setup_logger
from vlm_retrieval.agentic_pipeline import AgenticPlannerPipeline
from vlm_retrieval.config import VLMRetrievalConfig
from vlm_retrieval.frame_extractor import FrameExtractor
from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.search import RetrievalEngine
from vlm_retrieval.retrieval.vector_store import VectorStore
from vlm_retrieval.vqa import get_vqa_reasoner


def run_vlm_retrieval(config: VLMRetrievalConfig, query_text: str | None = None) -> None:
    logger = setup_logger("VLMRetrieval")
    logger.info("Starting Agentic VLM Retrieval Engine")
    logger.info(f"Retrieval model: {config.retrieval_model}")
    logger.info(f"Reasoning model: {config.reasoning_model}")
    logger.info(f"NPZ Dir/Path: {config.npz_dir or config.npz_path}")
    logger.info(f"Tracks JSON Path: {config.json_path}")
    logger.info(f"Video sources: {config.video_sources}")

    logger.info("Connecting to VectorStore via NPZ & Tracks JSON...")
    vector_store = VectorStore(
        npz_dir=config.npz_dir,
        npz_path=config.npz_path,
        json_path=config.json_path,
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
        device_map=config.device_map,
    )

    logger.info("Building Agentic Planner Pipeline...")
    pipeline = AgenticPlannerPipeline(
        retrieval_engine=retrieval_engine,
        vector_store=vector_store,
        frame_extractor=frame_extractor,
        reasoner=reasoner,
        max_planning_steps=config.max_planning_steps,
    )

    if query_text:
        logger.info(f"Executing query: '{query_text}'")
        results = pipeline.query(query_text=query_text, top_k=config.retrieval_top_k)
        logger.info(f"Retrieved {len(results)} ranked candidates:")
        for idx, res in enumerate(results, start=1):
            print(
                f"[{idx}] Camera: {res.camera_id} | Track ID: {res.global_id} | "
                f"Pos MS: {res.video_pos_ms:.1f} | Score: {res.vlm_score:.4f} | Explanation: {res.vlm_explanation}"
            )
    else:
        logger.info("Entering interactive query loop. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_q = input("\nEnter search query > ").strip()
                if user_q.lower() in ("exit", "quit"):
                    break
                if not user_q:
                    continue
                results = pipeline.query(query_text=user_q, top_k=config.retrieval_top_k)
                print(f"\n--- Results ({len(results)}) ---")
                for idx, res in enumerate(results, start=1):
                    print(
                        f"[{idx}] Camera: {res.camera_id} | Track ID: {res.global_id} | "
                        f"Pos MS: {res.video_pos_ms:.1f} | Score: {res.vlm_score:.4f} | Explanation: {res.vlm_explanation}"
                    )
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CCTV Agentic Planning VLM Retrieval Engine")
    parser.add_argument("--query", type=str, default=None, help="Natural language query to execute.")
    parser.add_argument("--npz_dir", type=str, default=None, help="Directory containing .npz embedding files.")
    parser.add_argument("--npz_path", type=str, default=None, help="Path to single .npz embeddings file.")
    parser.add_argument("--json_path", type=str, default=None, help="Path to accompanying tracks .json metadata file.")
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-so400m-patch14-384",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Options: 'Qwen/Qwen3-VL-8B-Instruct', 'openai-5.6', 'gemini-3.5-flash'",
    )
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--max_planning_steps", type=int, default=5)
    parser.add_argument(
        "--metadata_filter_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--device_map",
        type=str,
        default="balanced",
        help="Multi-GPU allocation strategy for local VLM (default: 'balanced').",
    )
    parser.add_argument(
        "--video_sources",
        type=str,
        default="{}",
        help='JSON dict mapping camera_id to video path, e.g. \'{"cam_1": "/app/dataset/video.avi"}\'',
    )
    args = parser.parse_args()

    config = VLMRetrievalConfig(
        npz_dir=args.npz_dir,
        npz_path=args.npz_path,
        json_path=args.json_path,
        retrieval_model=args.retrieval_model,
        reasoning_model=args.reasoning_model,
        video_sources=json.loads(args.video_sources),
        retrieval_top_k=args.retrieval_top_k,
        max_planning_steps=args.max_planning_steps,
        metadata_filter_enabled=args.metadata_filter_enabled,
        device=args.device,
        device_map=args.device_map,
    )
    run_vlm_retrieval(config, query_text=args.query)
