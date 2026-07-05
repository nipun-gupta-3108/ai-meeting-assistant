import os

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Load environment variables from .env
load_dotenv()

# Read Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env")


def load_pipeline():
    """
    Load the pretrained Speaker Diarization pipeline.
    """

    print("Loading Speaker Diarization Pipeline...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = load_pipeline()
    print("Pipeline Loaded Successfully!")