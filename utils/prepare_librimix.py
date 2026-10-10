"""Prepare static LibriMix CSVs for the original Mamba-TasNet data loader.

Uses the lab recipe's train-100/dev/test and mix_clean/s1/s2 layout while
leaving the upstream WSJ0Mix preparation unchanged. Input audio is read-only.
"""

import csv
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def prepare_librimix(
    datapath, savepath, n_spks=2, skip_prep=False, fs=8000,
    train_split="train-100", version="min", validate_audio=True,
):
    """Write libri{n_spks}mix_{split}.csv under savepath, never under datapath."""
    if skip_prep:
        return
    if n_spks not in (2, 3):
        raise ValueError("LibriMix supports 2 or 3 speakers")
    if fs not in (8000, 16000):
        raise ValueError("LibriMix sample rate must be 8000 or 16000")
    if train_split not in ("train-100", "train-360"):
        raise ValueError("train_split must be train-100 or train-360")
    if version not in ("min", "max"):
        raise ValueError("version must be min or max")
    root = Path(datapath).expanduser().resolve() / f"wav{fs // 1000}k" / version
    destination = Path(savepath)
    sources = [f"s{i}" for i in range(1, n_spks + 1)]
    fields = ["ID", "duration"]
    for name in ["mix"] + sources:
        fields.extend([f"{name}_wav", f"{name}_wav_format", f"{name}_wav_opts"])

    # Validate all partitions before writing any CSV.
    partitions = []
    for split in (train_split, "dev", "test"):
        folder = root / split
        lengths = {}
        if not validate_audio:
            metadata = root / "metadata" / f"mixture_{split}_mix_clean.csv"
            print(f"[DATA] Reading {metadata}", flush=True)
            with metadata.open(newline="", encoding="utf-8") as stream:
                for item in csv.DictReader(stream):
                    mixture = folder / "mix_clean" / Path(item["mixture_path"]).name
                    frames = int(item["length"])
                    if frames < 1 or mixture in lengths:
                        raise ValueError(f"Invalid length or duplicate mixture in {metadata}")
                    lengths[mixture] = frames
            mixtures = sorted(lengths)
        else:
            print(f"[DATA] Listing and validating audio in {folder}", flush=True)
            mixtures = sorted((folder / "mix_clean").glob("*.wav"))
        if not mixtures:
            raise FileNotFoundError(f"No WAV mixtures found in {folder / 'mix_clean'}")
        rows = []
        for index, mixture in enumerate(tqdm(
            mixtures, desc=f"Prepare LibriMix {split}", mininterval=2,
        )):
            paths = [mixture] + [folder / source / mixture.name for source in sources]
            # Fast mode reads official lengths and checks one audio pair per
            # split; the normal data loader still reads actual WAVs at train.
            if validate_audio or index == 0:
                infos = [sf.info(str(path)) for path in paths]
                for path, info in zip(paths, infos):
                    if info.samplerate != fs or info.channels != 1:
                        raise ValueError(f"Expected mono {fs} Hz audio: {path}")
                    if info.frames != infos[0].frames:
                        raise ValueError(f"Mixture/source lengths differ: {path}")
                if not validate_audio and infos[0].frames != lengths[mixture]:
                    raise ValueError(f"Metadata/audio lengths differ: {mixture}")
            frames = infos[0].frames if validate_audio else lengths[mixture]
            row = {"ID": mixture.stem, "duration": frames / fs}
            for name, path in zip(["mix"] + sources, paths):
                row.update({f"{name}_wav": str(path), f"{name}_wav_format": "wav",
                            f"{name}_wav_opts": ""})
            rows.append(row)
        partitions.append((split, rows))

    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in partitions:
        with (destination / f"libri{n_spks}mix_{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[DATA] {split}: {len(rows)} mixtures -> {stream.name}", flush=True)
