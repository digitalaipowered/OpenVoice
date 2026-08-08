from __future__ import annotations

import os
import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = Path(os.getenv("OPENVOICE_CKPT_DIR", ROOT / "checkpoints_v2"))
OUTPUT_DIR = Path(os.getenv("OPENVOICE_FORGE_OUTPUT_DIR", ROOT / "outputs_forge"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = {
    "English": "EN_NEWEST",
    "Spanish": "ES",
    "French": "FR",
    "Chinese": "ZH",
    "Japanese": "JP",
    "Korean": "KR",
}

MAX_TEXT_CHARS = int(os.getenv("OPENVOICE_FORGE_MAX_TEXT_CHARS", "1800"))
MIN_REFERENCE_SECONDS = float(os.getenv("OPENVOICE_FORGE_MIN_REFERENCE_SECONDS", "2"))
MAX_REFERENCE_SECONDS = float(os.getenv("OPENVOICE_FORGE_MAX_REFERENCE_SECONDS", "60"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_CONVERTER_LOCK = threading.Lock()


def _require_checkpoint(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Missing {label}: {path}. Download and extract the official OpenVoice V2 checkpoints before starting Forge."
        )


@lru_cache(maxsize=1)
def get_converter() -> ToneColorConverter:
    config_path = CHECKPOINT_DIR / "converter" / "config.json"
    checkpoint_path = CHECKPOINT_DIR / "converter" / "checkpoint.pth"
    _require_checkpoint(config_path, "converter config")
    _require_checkpoint(checkpoint_path, "converter checkpoint")
    converter = ToneColorConverter(str(config_path), device=DEVICE)
    converter.load_ckpt(str(checkpoint_path))
    return converter


@lru_cache(maxsize=len(LANGUAGES))
def get_tts(language_code: str) -> TTS:
    return TTS(language=language_code, device=DEVICE)


def _duration_seconds(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


def _pick_base_speaker(model: TTS) -> tuple[str, int, Path]:
    speaker_ids = model.hps.data.spk2id
    if not speaker_ids:
        raise RuntimeError("MeloTTS returned no base speakers for the selected language.")

    speaker_key = next(iter(speaker_ids.keys()))
    speaker_id = int(speaker_ids[speaker_key])
    normalized_key = speaker_key.lower().replace("_", "-")
    source_embedding = CHECKPOINT_DIR / "base_speakers" / "ses" / f"{normalized_key}.pth"
    _require_checkpoint(source_embedding, f"base speaker embedding for {speaker_key}")
    return speaker_key, speaker_id, source_embedding


def synthesize(
    reference_audio: Optional[str],
    text: str,
    language_label: str,
    speed: float,
    rights_confirmed: bool,
) -> tuple[str, str]:
    if not rights_confirmed:
        raise gr.Error("Confirm that you own the voice or have permission to clone it.")
    if not reference_audio:
        raise gr.Error("Upload a reference voice clip first.")

    clean_text = (text or "").strip()
    if not clean_text:
        raise gr.Error("Enter text to synthesize.")
    if len(clean_text) > MAX_TEXT_CHARS:
        raise gr.Error(f"Text is limited to {MAX_TEXT_CHARS} characters per generation.")
    if language_label not in LANGUAGES:
        raise gr.Error("Choose a supported language.")

    try:
        duration = _duration_seconds(reference_audio)
    except Exception as exc:
        raise gr.Error(f"Could not read the reference audio: {exc}") from exc

    if duration < MIN_REFERENCE_SECONDS:
        raise gr.Error(f"Reference audio must be at least {MIN_REFERENCE_SECONDS:g} seconds long.")
    if duration > MAX_REFERENCE_SECONDS:
        raise gr.Error(f"Reference audio must be no longer than {MAX_REFERENCE_SECONDS:g} seconds.")

    language_code = LANGUAGES[language_label]
    converter = get_converter()
    model = get_tts(language_code)
    speaker_key, speaker_id, source_embedding_path = _pick_base_speaker(model)

    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"openvoice-forge-{job_id[:8]}-"))
    source_wav = work_dir / "base.wav"
    output_wav = OUTPUT_DIR / f"forge-{job_id}.wav"

    try:
        with _CONVERTER_LOCK:
            target_se, _ = se_extractor.get_se(reference_audio, converter, vad=True)
            source_se = torch.load(str(source_embedding_path), map_location=DEVICE)

            if torch.backends.mps.is_available() and DEVICE == "cpu":
                torch.backends.mps.is_available = lambda: False

            model.tts_to_file(clean_text, speaker_id, str(source_wav), speed=float(speed))
            converter.convert(
                audio_src_path=str(source_wav),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_wav),
                message="@OpenVoiceForge",
            )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc

    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise gr.Error("Generation completed without a usable output file.")

    backend = "GPU" if DEVICE.startswith("cuda") else "CPU"
    status = f"Generated with OpenVoice V2 + MeloTTS on {backend}. Base speaker: {speaker_key}."
    return str(output_wav), status


def build_demo() -> gr.Blocks:
    css = """
    .forge-wrap {max-width: 980px; margin: 0 auto;}
    .forge-hero {padding: 18px 0 8px 0;}
    .forge-title {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em;}
    .forge-sub {opacity: .78; max-width: 760px; line-height: 1.55;}
    .forge-note {font-size: .9rem; opacity: .75;}
    """

    with gr.Blocks(title="OpenVoice Forge", css=css) as demo:
        with gr.Column(elem_classes=["forge-wrap"]):
            gr.HTML(
                "<div class='forge-hero'><div class='forge-title'>OpenVoice Forge</div>"
                "<div class='forge-sub'>Instant voice cloning powered by your OpenVoice V2 fork. "
                "Upload a reference clip, choose a supported language, and generate a downloadable WAV using the real OpenVoice tone-color conversion pipeline.</div></div>"
            )

            with gr.Row():
                reference = gr.Audio(
                    label="Reference voice",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                output = gr.Audio(label="Generated voice", type="filepath")

            text = gr.Textbox(
                label="What should the cloned voice say?",
                placeholder="Type the speech you want to generate...",
                lines=6,
                max_lines=12,
            )

            with gr.Row():
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Language",
                )
                speed = gr.Slider(
                    minimum=0.7,
                    maximum=1.35,
                    value=1.0,
                    step=0.05,
                    label="Speech speed",
                )

            rights = gr.Checkbox(
                label="I own this voice or have explicit permission to clone and synthesize it.",
                value=False,
            )
            generate = gr.Button("Generate cloned voice", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            gr.HTML(
                "<div class='forge-note'>Reference clips are processed by the running OpenVoice instance. "
                "For public deployments, keep access controlled and use only voices you are authorized to clone.</div>"
            )

            generate.click(
                fn=synthesize,
                inputs=[reference, text, language, speed, rights],
                outputs=[output, status],
            )

    return demo


def launch(share: bool = False) -> None:
    username = os.getenv("OPENVOICE_FORGE_USERNAME")
    password = os.getenv("OPENVOICE_FORGE_PASSWORD")
    auth = (username, password) if username and password else None
    port = int(os.getenv("PORT", os.getenv("OPENVOICE_FORGE_PORT", "7860")))
    demo = build_demo()
    demo.queue(concurrency_count=1, max_size=12).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        auth=auth,
    )
