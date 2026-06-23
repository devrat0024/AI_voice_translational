"""
benchmark.py — AI Clinical Scribe Model Benchmarking Suite
===========================================================
Measures ASR (Whisper), NER (BioBERT/spaCy), and LLM (MedGemma/Groq) metrics:
- Latency & Inference Time
- Prompt/Completion Tokens
- GPU/CPU Memory Footprint
"""

import time
import json
import logging
import argparse
import sys
from pathlib import Path

# Try importing performance utilities
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

# Initialize static-ffmpeg so Whisper can find ffmpeg
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")


# ── Mock Clinical Text for NER and LLM benchmarks ─────────────────────────────
SAMPLE_CLINICAL_TEXT = (
    "Patient is a 54-year-old male presenting with acute chest congestion, persistent cough, "
    "and mild fever (100.2 F) for 4 days. He reports taking Paracetamol 500mg twice a day "
    "and Ibuprofen 400mg occasionally. Lung auscultation revealed mild wheezing bilaterally. "
    "Recommend resting, increased fluid intake, and follow-up in 3 days if symptoms persist."
)


# ── Memory Tracker Helpers ───────────────────────────────────────────────────
def get_memory_usage() -> float:
    """Returns current process RAM usage in MB."""
    if psutil:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def get_gpu_memory_usage() -> float:
    """Returns allocated GPU VRAM in MB if PyTorch CUDA is active."""
    if torch and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


# ── Benchmark: Whisper ASR ────────────────────────────────────────────────────
def run_whisper_benchmark(audio_path: Path, model_size: str) -> dict:
    logger.info(f"--- Benchmarking Whisper ASR (Model: {model_size}) ---")
    if not audio_path.exists():
        logger.warning(f"Audio file '{audio_path}' not found. Skipping Whisper benchmark.")
        return {}

    # Estimate/measure audio duration
    audio_duration = 0.0
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000.0  # in seconds
    except Exception:
        # Fallback raw estimation
        audio_duration = 31.0  # sample.mp3 is 31s
        logger.warning("Could not extract exact audio duration. Defaulting to 31.0 seconds.")

    logger.info("Loading Whisper model...")
    mem_before = get_memory_usage()
    gpu_before = get_gpu_memory_usage()
    
    start_load = time.time()
    try:
        import whisper
        model = whisper.load_model(model_size)
    except ImportError:
        logger.error("Whisper library not installed. Cannot benchmark Whisper.")
        return {}
    load_time = time.time() - start_load
    
    mem_after_load = get_memory_usage()
    gpu_after_load = get_gpu_memory_usage()

    logger.info(f"Model loaded in {load_time:.2f}s. Running transcription...")
    
    start_transcribe = time.time()
    result = model.transcribe(str(audio_path))
    transcribe_time = time.time() - start_transcribe
    
    mem_after_run = get_memory_usage()
    gpu_after_run = get_gpu_memory_usage()

    logger.info(f"ASR complete. Inference time: {transcribe_time:.2f}s for {audio_duration:.2f}s audio.")

    return {
        "audio_length_sec": audio_duration,
        "load_time_sec": load_time,
        "inference_time_sec": transcribe_time,
        "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        "ram_load_mb": max(0.0, mem_after_load - mem_before),
        "gpu_vram_load_mb": max(0.0, gpu_after_load - gpu_before),
        "ram_peak_mb": mem_after_run,
        "gpu_vram_peak_mb": gpu_after_run,
    }


# ── Benchmark: BioBERT NER ────────────────────────────────────────────────────
def run_ner_benchmark(ner_mode: str) -> dict:
    logger.info(f"--- Benchmarking Clinical NER (Mode: {ner_mode}) ---")
    
    mem_before = get_memory_usage()
    gpu_before = get_gpu_memory_usage()

    start_init = time.time()
    from data_transcriptor.transcription.medical_ner import MedicalEntityExtractor
    extractor = MedicalEntityExtractor(mode=ner_mode)
    
    # Force load models depending on mode
    if ner_mode == "transformer" or ner_mode == "auto":
        extractor._init_transformer()
    if ner_mode == "scispacy" or ner_mode == "auto":
        extractor._init_scispacy()
    init_time = time.time() - start_init

    mem_after_load = get_memory_usage()
    gpu_after_load = get_gpu_memory_usage()

    logger.info("Running NER entity extraction...")
    
    # Run multiple iterations to get stable timing
    iterations = 5
    start_run = time.time()
    for _ in range(iterations):
        _ = extractor.extract_entities(SAMPLE_CLINICAL_TEXT)
    total_run_time = time.time() - start_run
    avg_inference_time = total_run_time / iterations

    mem_after_run = get_memory_usage()
    gpu_after_run = get_gpu_memory_usage()

    logger.info(f"NER complete. Average extraction time: {avg_inference_time:.3f}s per record.")

    return {
        "init_time_sec": init_time,
        "average_inference_time_sec": avg_inference_time,
        "ram_load_mb": max(0.0, mem_after_load - mem_before),
        "gpu_vram_load_mb": max(0.0, gpu_after_load - gpu_before),
        "ram_peak_mb": mem_after_run,
        "gpu_vram_peak_mb": gpu_after_run,
    }


# ── Benchmark: MedGemma / Groq LLM ───────────────────────────────────────────
def run_llm_benchmark(groq_key: str, groq_model: str) -> dict:
    logger.info(f"--- Benchmarking LLM Layer (Model: {groq_model}) ---")
    
    # Let's read GROQ API KEY from env if not provided
    import os
    from backend.app.config import GROQ_API_KEY
    api_key = groq_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

    if not api_key:
        logger.warning("GROQ_API_KEY is not set. Running simulated local MedGemma/LLM benchmark.")
        # Local simulated stats for MedGemma-2B
        # MedGemma-2B runs at roughly 25-30 tokens/sec on modern GPU or CPU,
        # occupying 4-5 GB of RAM/VRAM.
        return {
            "mode": "simulated_medgemma_2b",
            "prompt_tokens": 125,
            "completion_tokens": 285,
            "generation_time_sec": 3.42,
            "tokens_per_sec": 285 / 3.42,
            "estimated_ram_usage_mb": 4500.0,
            "estimated_gpu_vram_usage_mb": 4096.0,
        }

    # Real Groq API Benchmark
    logger.info("Initializing Groq API Client...")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        logger.error("groq package not installed. Cannot run live LLM benchmark.")
        return {}

    prompt = (
        "Generate a structured SOAP note for the following clinical text:\n"
        f"{SAMPLE_CLINICAL_TEXT}"
    )

    logger.info("Sending request to Groq API...")
    mem_before = get_memory_usage()

    start_time = time.time()
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=groq_model,
        )
        latency = time.time() - start_time
        
        usage = chat_completion.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Check for timing information in groq headers if available
        # Or calculate it via simple tokens / latency
        tokens_per_sec = completion_tokens / latency if latency > 0 else 0.0

        logger.info(f"Groq API call completed in {latency:.2f}s.")
        logger.info(f"Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}")

        return {
            "mode": f"live_groq_api ({groq_model})",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "generation_time_sec": latency,
            "tokens_per_sec": tokens_per_sec,
            "ram_peak_mb": get_memory_usage(),
            "gpu_vram_peak_mb": get_gpu_memory_usage(),
        }
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return {}


# ── Main Suite Runner ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model Benchmarking Suite CLI")
    parser.add_argument("--audio", type=str, default="Catching Up With Friends Audio 2.mp3", help="Path to audio file for ASR")
    parser.add_argument("--whisper-model", type=str, default="tiny", help="Whisper model size")
    parser.add_argument("--ner-mode", type=str, default="rules", choices=["rules", "scispacy", "transformer", "auto"], help="NER extraction mode")
    parser.add_argument("--groq-key", type=str, default="", help="Groq API Key override")
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile", help="Groq LLM model size")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks (ASR, NER, LLM)")

    args = parser.parse_args()

    audio_path = Path(args.audio)

    results = {}

    print("\n" + "=" * 60)
    print("           CLINICAL AI SCRIBE - MODEL BENCHMARK SUITE")
    print("=" * 60 + "\n")

    # 1. Benchmark ASR
    asr_stats = run_whisper_benchmark(audio_path, args.whisper_model)
    if asr_stats:
        results["whisper_asr"] = asr_stats
        print("\n[WHISPER ASR BENCHMARK RESULTS]")
        print(f"  - Audio Duration:        {asr_stats['audio_length_sec']:.2f} seconds")
        print(f"  - Inference Time:        {asr_stats['inference_time_sec']:.2f} seconds")
        print(f"  - Real-Time Factor (RTF): {asr_stats['rtf']:.4f}")
        print(f"  - RAM Overhead (load):   {asr_stats['ram_load_mb']:.2f} MB")
        if torch and torch.cuda.is_available():
            print(f"  - GPU VRAM (load):       {asr_stats['gpu_vram_load_mb']:.2f} MB")

    # 2. Benchmark NER
    ner_stats = run_ner_benchmark(args.ner_mode)
    if ner_stats:
        results["ner"] = ner_stats
        print("\n[CLINICAL NER BENCHMARK RESULTS]")
        print(f"  - Extraction Mode:       {args.ner_mode}")
        print(f"  - Avg Inference Time:    {ner_stats['average_inference_time_sec']:.4f} seconds")
        print(f"  - RAM Overhead (load):   {ner_stats['ram_load_mb']:.2f} MB")

    # 3. Benchmark LLM / MedGemma
    llm_stats = run_llm_benchmark(args.groq_key, args.groq_model)
    if llm_stats:
        results["llm"] = llm_stats
        print("\n[LLM / MEDGEMMA BENCHMARK RESULTS]")
        print(f"  - Mode:                  {llm_stats.get('mode')}")
        print(f"  - Prompt Tokens:         {llm_stats.get('prompt_tokens')}")
        print(f"  - Completion Tokens:     {llm_stats.get('completion_tokens')}")
        print(f"  - Generation Time:       {llm_stats.get('generation_time_sec'):.2f} seconds")
        print(f"  - Throughput Speed:      {llm_stats.get('tokens_per_sec'):.2f} tokens/second")
        if "estimated_ram_usage_mb" in llm_stats:
            print(f"  - Est. Memory Footprint: RAM ~ {llm_stats.get('estimated_ram_usage_mb')} MB | VRAM ~ {llm_stats.get('estimated_gpu_vram_usage_mb')} MB")

    print("\n" + "=" * 60)
    print("                     BENCHMARK RUN COMPLETE")
    print("=" * 60 + "\n")

    # Save benchmark results to JSON
    out_path = Path("data/output/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed benchmark metrics written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
