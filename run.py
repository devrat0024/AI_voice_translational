"""
run.py — Unified AI Clinical Scribe Entry Point
================================================

Usage:
  python run.py api                        Start the FastAPI server (default: http://127.0.0.1:8000)
  python run.py cli [options]              Run CLI transcription pipeline on an audio file
  python run.py etl <command>              Run ETL data pipeline commands

API Server Options:
  --host HOST          Server host (default: 127.0.0.1, env: HOST)
  --port PORT          Server port (default: 8000, env: PORT)
  --reload             Enable auto-reload for development

CLI Transcription Options:
  --audio PATH         Path to clinical audio file
  --whisper-model      Whisper model size: tiny|base|small|medium|large (default: tiny)
  --ner-mode           NER mode: scispacy|transformer|rules|auto (default: auto)
  --hf-token TOKEN     Hugging Face token for pyannote.audio speaker diarization
  --groq-key KEY       Groq API key (falls back to GROQ_API_KEY env variable)
  --groq-model MODEL   Groq LLM model name (default: llama-3.3-70b-versatile)
  --output PATH        Output JSON file path (default: data/output/<audio_name>_structured.json)

ETL Commands:
  init-db              Initialize ETL database and directory structure
  run-etl              Ingest and process all files in data/raw/
  show-stats           Show ETL database record counts
  query "SQL"          Execute a raw SQL query on the ETL database
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run")


def run_api(args):
    """Starts the FastAPI/Uvicorn server."""
    import os
    import uvicorn

    host = args.host or os.getenv("HOST", "127.0.0.1")
    port = args.port or int(os.getenv("PORT", "8000"))
    reload = args.reload or os.getenv("ENV", "development") == "development"

    print(f"\n🚀  Starting Clinical AI Scribe API at http://{host}:{port}")
    print(f"📖  Swagger UI available at http://{host}:{port}/docs\n")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=reload)


def run_cli(args):
    """Runs the full structured clinical pipeline from the command line."""
    from backend.app.config import HF_TOKEN
    from data_transcriptor.transcription.runner import ClinicalPipeline
    from data_transcriptor.transcription.schemas import PipelineConfig

    # Resolve audio file path
    audio_path = None
    if args.audio:
        audio_path = Path(args.audio)
    else:
        default_locations = [
            Path("Catching Up With Friends Audio 2.mp3"),
            Path("sample.mp3"),
            Path("data/raw/audio/sample.mp3"),
        ]
        for loc in default_locations:
            if loc.exists():
                audio_path = loc
                logger.info(f"No --audio specified. Using sample: {audio_path.name}")
                break

        if not audio_path:
            logger.error(
                "No audio file specified and no sample audio found.\n"
                "Use: python run.py cli --audio <path_to_audio_file>"
            )
            sys.exit(1)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path.resolve()}")
        sys.exit(1)

    config = PipelineConfig(
        whisper_model=args.whisper_model,
        ner_mode=args.ner_mode,
        hf_token=args.hf_token or HF_TOKEN,
        groq_api_key=args.groq_key,
        groq_model=args.groq_model,
        output_path=args.output,
    )

    try:
        pipeline = ClinicalPipeline(config)
        result = pipeline.run(audio_path)

        summary = result.summary_dict()
        print("\n" + "=" * 60)
        print("PIPELINE RESULT SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

        if result.transcription:
            print("\n" + "=" * 60)
            print("DIARIZED TRANSCRIPT")
            print("=" * 60)
            for turn in result.transcription.dialogue:
                print(f"[{turn.start:.2f}s - {turn.end:.2f}s] {turn.speaker}: {turn.text}")

        if result.clinical_intelligence:
            print("\n" + "=" * 60)
            print("SOAP NOTE")
            print("=" * 60)
            print(result.clinical_intelligence.soap_note.raw)
            print("\n" + "=" * 60)
            print("CLINICAL SUMMARY")
            print("=" * 60)
            print(result.clinical_intelligence.clinical_summary)

        print("\n" + "=" * 60)
        print(f"Full structured JSON saved.")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def run_etl(args):
    """Runs ETL pipeline commands."""
    from data_pipeline.pipeline.config import init_directories, DB_PATH
    from data_pipeline.pipeline.storage import init_db, get_connection, get_stats
    from data_pipeline.pipeline.etl import run_etl_pipeline

    if args.etl_command == "init-db":
        logger.info("Initializing ETL directories...")
        init_directories()
        logger.info("Initializing ETL database...")
        init_db()
        logger.info("ETL setup complete.")

    elif args.etl_command == "run-etl":
        logger.info("Starting ETL Pipeline run...")
        stats = run_etl_pipeline()
        print("\n=== ETL RUN COMPLETED ===")
        print(f"Total Patients:         {stats.get('patients_count', 0)}")
        print(f"Audio Records (counts): {stats.get('audio_counts', {})}")
        print(f"Documents (counts):     {stats.get('document_counts', {})}")
        print("=========================")

    elif args.etl_command == "show-stats":
        if not DB_PATH.exists():
            logger.error("Database not initialized. Run: python run.py etl init-db")
            sys.exit(1)
        conn = get_connection()
        try:
            stats = get_stats(conn)
            print(f"\n=== ETL DATABASE STATISTICS ===")
            print(f"Database: {DB_PATH.resolve()}")
            print(f"Total Patients: {stats.get('patients_count', 0)}")
            print("\nAudio Records:")
            for status, count in stats.get("audio_counts", {}).items():
                print(f"  {status}: {count}")
            print("\nMedical Documents:")
            for status, count in stats.get("document_counts", {}).items():
                print(f"  {status}: {count}")
            print("================================")
        finally:
            conn.close()

    elif args.etl_command == "query":
        if not DB_PATH.exists():
            logger.error("Database not initialized. Run: python run.py etl init-db")
            sys.exit(1)
        import pandas as pd
        conn = get_connection()
        try:
            df = pd.read_sql_query(args.sql, conn)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            print("\n=== QUERY RESULTS ===")
            print(df.to_string())
            print("=====================")
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
        finally:
            conn.close()

    else:
        print("ETL commands: init-db | run-etl | show-stats | query \"SQL\"")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="AI Clinical Scribe — Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # api sub-command
    api_parser = subparsers.add_parser("api", help="Start the FastAPI REST API server")
    api_parser.add_argument("--host", type=str, default="", help="Server host (default: 127.0.0.1)")
    api_parser.add_argument("--port", type=int, default=0, help="Server port (default: 8000)")
    api_parser.add_argument("--reload", action="store_true", help="Enable hot-reload")

    # cli sub-command
    cli_parser = subparsers.add_parser("cli", help="Run CLI transcription pipeline")
    cli_parser.add_argument("--audio", type=str, default="", help="Path to audio file")
    cli_parser.add_argument("--whisper-model", type=str, default="tiny",
                             choices=["tiny", "base", "small", "medium", "large"])
    cli_parser.add_argument("--ner-mode", type=str, default="auto",
                             choices=["scispacy", "transformer", "rules", "auto"])
    cli_parser.add_argument("--hf-token", type=str, default="", help="Hugging Face token")
    cli_parser.add_argument("--groq-key", type=str, default="", help="Groq API key")
    cli_parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile",
                             help="Groq model name")
    cli_parser.add_argument("--output", type=str, default="", help="Output JSON file path")

    # etl sub-command
    etl_parser = subparsers.add_parser("etl", help="Run ETL data pipeline commands")
    etl_subparsers = etl_parser.add_subparsers(dest="etl_command")
    etl_subparsers.add_parser("init-db", help="Initialize ETL database and directories")
    etl_subparsers.add_parser("run-etl", help="Ingest and process all files in data/raw/")
    etl_subparsers.add_parser("show-stats", help="Show ETL database record counts")
    query_p = etl_subparsers.add_parser("query", help="Run a raw SQL query on the ETL database")
    query_p.add_argument("sql", type=str, help="SQL query string")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "api":
        run_api(args)
    elif args.mode == "cli":
        run_cli(args)
    elif args.mode == "etl":
        if not args.etl_command:
            parser.parse_args(["etl", "--help"])
        else:
            run_etl(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
