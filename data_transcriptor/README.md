# Clinical Speech to Structured Medical Information Pipeline

An end-to-end AI/ML engineering pipeline that transcribes raw clinical conversations, performs speaker diarization, aligns audio segments, extracts medical entities (symptoms and medicines), and saves structured medical records in JSON.

---

## Pipeline Workflow

1. **Speech Recognition**: Uses OpenAI Whisper to convert clinical conversations into timestamped audio segments.
2. **Speaker Diarization**: Leverages `pyannote.audio` (via pre-trained gated pipelines) to identify speaker turns (e.g. Doctor vs. Patient). Falls back to transcription dialogue turns if pyannote token/model is not supplied.
3. **Alignment**: Map speakers to transcription text turns using segment-overlap algorithms.
4. **Medical NER (Named Entity Recognition)**: Extracts key entities (symptoms & medicines) utilizing BioBERT (`AlText/clinical-ner` Hugging Face model), SciSpacy (`en_core_sci_sm`), or a fallback regex keyword mapper.

---

## Setup & Dependencies

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Hugging Face Token Setup (Optional for Pyannote)**:
   For speaker diarization, sign the user agreement on Hugging Face for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), then set the token as an environment variable:
   ```bash
   $env:HF_TOKEN="your_huggingface_token"
   ```

---

## Running the Pipeline

To run the pipeline on an audio file:
```bash
python main.py --audio "path/to/clinical_session.mp3"
```

### Specifying Models and Outputs
To customize the Whisper model, NER pipeline mode, or specify the output JSON path:
```bash
python main.py --audio "path/to/clinical_session.mp3" --whisper-model "base" --ner-mode "auto" --output "output_data.json"
```

---

## Output Schema

The output will yield the following JSON structure:
```json
{
  "symptom": "Fever",
  "medicine": "Paracetamol"
}
```
Saved automatically under `data/output/<audio_filename>_structured.json`.
