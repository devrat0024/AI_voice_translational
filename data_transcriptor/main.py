import argparse
import json
import logging
import sys
from pathlib import Path
from transcription.config import init_directories, OUTPUT_DIR, HF_TOKEN
from transcription.pipeline import ClinicalTranscriptionPipeline

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
    parser = argparse.ArgumentParser(description="AI Scribe Speech to Structured Medical Information Pipeline CLI")
    parser.add_argument("--audio", type=str, help="Path to clinical audio file (optional fallback to workspace samples)")
    parser.add_argument("--whisper-model", type=str, default="tiny", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--ner-mode", type=str, default="auto", choices=["scispacy", "transformer", "rules", "auto"], help="Medical NER extraction mode")
    parser.add_argument("--hf-token", type=str, default="", help="Hugging Face User Access Token for pyannote.audio")
    parser.add_argument("--groq-key", type=str, default="", help="Groq API Key (falls back to GROQ_API_KEY env variable)")
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile", help="Groq LLM model name")
    parser.add_argument("--output", type=str, default="", help="Path to write output JSON (default: data/output/<audio_name>_structured.json)")

    args = parser.parse_args()

    init_directories()

    audio_path = None
    if args.audio:
        audio_path = Path(args.audio)
    else:
        # Fallback search for default sample files in the workspace
        default_locations = [
            Path("../AI_voice_translational/Catching Up With Friends Audio 2.mp3"),
            Path("../AI_voice_translational/sample.mp3"),
            Path("data/raw/Catching Up With Friends Audio 2.mp3")
        ]
        for loc in default_locations:
            if loc.exists():
                audio_path = loc
                logger.info(f"No --audio argument specified. Defaulting to sample audio: {audio_path.name}")
                break
        
        if not audio_path:
            logger.error("No audio file specified, and no default sample audio files could be located in the workspace.")
            parser.print_help()
            sys.exit(1)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path.resolve()}")
        sys.exit(1)

    # Use CLI arg token first, fallback to env variable HF_TOKEN
    token = args.hf_token or HF_TOKEN

    # Execute pipeline
    logger.info("Initializing Clinical Transcription & NER Pipeline...")
    pipeline = ClinicalTranscriptionPipeline(
        whisper_model=args.whisper_model,
        ner_mode=args.ner_mode,
        hf_token=token
    )

    try:
        results = pipeline.run_pipeline(audio_path)
        
        # Display Aligned Dialogue
        print("\n" + "=" * 50)
        print("DIARIZED TRANSCRIPT:")
        print("=" * 50)
        for turn in results["dialogue"]:
            print(f"[{turn['start']:.2f}s - {turn['end']:.2f}s] {turn['speaker']}: {turn['text']}")
        print("=" * 50)

        # Display Structured Medical Info
        structured_info = results["structured_info"]
        print("\nSTRUCTURED MEDICAL ENTITIES:")
        print(json.dumps(structured_info, indent=2))
        print("=" * 50)

        # Initialize Clinical Intelligence LLM Layer
        from transcription.llm_layer import ClinicalIntelligenceLayer
        logger.info("Initializing Clinical LLM Intelligence Layer (Groq)...")
        llm_layer = ClinicalIntelligenceLayer(
            model_name=args.groq_model,
            api_key=args.groq_key
        )

        # Perform Medical Spelling Correction Demonstration
        spell_demo_input = "The lab test shows elevated lymphosite and colestrol levels."
        spell_demo_output = llm_layer.medical_correction(spell_demo_input)

        # Perform operations on transcript
        corrected_transcript = llm_layer.medical_correction(results["full_text"])
        soap_note = llm_layer.generate_soap_note(results["full_text"])
        clinical_summary = llm_layer.generate_clinical_summary(results["full_text"])

        # Display LLM outputs
        print("\n" + "=" * 50)
        print("CLINICAL INTELLIGENCE LAYER (LLM):")
        print("=" * 50)
        print(f"Medical Spelling Correction Demonstration:")
        print(f"  Input:  '{spell_demo_input}'")
        print(f"  Output: '{spell_demo_output}'")
        print("-" * 50)
        print("Corrected Transcript Text:")
        print(corrected_transcript)
        print("-" * 50)
        print("Generated SOAP Clinical Note:")
        print(soap_note)
        print("-" * 50)
        print("Clinical Summary:")
        print(clinical_summary)
        print("=" * 50)

        # Save structured info and LLM outputs to JSON File
        output_file_path = Path(args.output) if args.output else OUTPUT_DIR / f"{audio_path.stem}_structured.json"
        
        structured_info["spelling_correction_demo"] = {
            "input": spell_demo_input,
            "output": spell_demo_output
        }
        structured_info["corrected_transcript"] = corrected_transcript
        structured_info["soap_note"] = soap_note
        structured_info["clinical_summary"] = clinical_summary

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(structured_info, f, indent=2)
        logger.info(f"Structured JSON output saved to: {output_file_path.resolve()}")

        # Output Transcript Text File
        transcript_file_path = output_file_path.with_name(f"{output_file_path.stem.replace('_structured', '')}_transcript.txt")
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("FULL CONVERSATION TRANSCRIPTION\n")
            f.write("=" * 60 + "\n")
            f.write(results["full_text"].strip() + "\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("DIARIZED CLINICAL TRANSCRIPT\n")
            f.write("=" * 60 + "\n")
            for turn in results["dialogue"]:
                f.write(f"[{turn['start']:.2f}s - {turn['end']:.2f}s] {turn['speaker']}: {turn['text']}\n")
            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("CLINICAL INTELLIGENCE LAYER OUTPUTS (GROQ)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Medical Spelling Correction Demonstration:\n")
            f.write(f"  Input:  {spell_demo_input}\n")
            f.write(f"  Output: {spell_demo_output}\n\n")
            f.write(f"Corrected Full Text:\n{corrected_transcript}\n\n")
            f.write(f"SOAP Note:\n{soap_note}\n\n")
            f.write(f"Clinical Summary:\n{clinical_summary}\n")
            f.write("=" * 60 + "\n")
        logger.info(f"Transcript text output saved to: {transcript_file_path.resolve()}")

    except Exception as e:
        logger.error(f"Pipeline run failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
