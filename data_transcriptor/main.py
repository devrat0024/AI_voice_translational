import argparse
import json
import logging
import sys
from pathlib import Path

from backend.app.config import init_directories, OUTPUT_DIR, HF_TOKEN
from data_transcriptor.transcription.runner import ClinicalPipeline
from data_transcriptor.transcription.schemas import PipelineConfig

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="Clinical Transcription & AI Scribe Pipeline CLI")
    parser.add_argument("--audio", type=str, help="Path to clinical audio file")
    parser.add_argument("--whisper-model", type=str, default="tiny", help="Whisper model size (tiny, base, etc.)")
    parser.add_argument("--ner-mode", type=str, default="auto", help="NER extraction mode")
    parser.add_argument("--hf-token", type=str, default="", help="Hugging Face User Access Token")
    parser.add_argument("--groq-key", type=str, default="", help="Groq API Key")
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile", help="Groq LLM model name")
    parser.add_argument("--output", type=str, default="", help="Path to write output JSON")

    args = parser.parse_args()

    init_directories()

    audio_path = None
    if args.audio:
        audio_path = Path(args.audio)
    else:
        # Fallback search for default sample files in the workspace
        default_locations = [
            Path("Catching Up With Friends Audio 2.mp3"),
            Path("sample.mp3"),
            Path("data/raw/Catching Up With Friends Audio 2.mp3")
        ]
        for loc in default_locations:
            if loc.exists():
                audio_path = loc
                logger.info(f"No --audio specified. Defaulting to sample audio: {audio_path.name}")
                break
        
        if not audio_path:
            logger.error("No audio file specified, and no default sample audio files could be located.")
            parser.print_help()
            sys.exit(1)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path.resolve()}")
        sys.exit(1)

    # Use CLI arg token first, fallback to config HF_TOKEN
    token = args.hf_token or HF_TOKEN

    # Set up config
    config = PipelineConfig(
        whisper_model=args.whisper_model,
        ner_mode=args.ner_mode,
        hf_token=token,
        groq_api_key=args.groq_key,
        groq_model=args.groq_model,
        output_path=args.output
    )

    logger.info("Initializing Structured Clinical AI Pipeline...")
    pipeline = ClinicalPipeline(config)
    
    try:
        result = pipeline.run(audio_path)
        
        # Display the result summary
        print("\n" + "=" * 60)
        print("PIPELINE RESULT SUMMARY")
        print("=" * 60)
        print(json.dumps(result.summary_dict(), indent=2))
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
