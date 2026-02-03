"""
Storybook Speech Splitter - ComfyUI Custom Node
Splits multi-speaker audio into per-speaker tracks.

Short clips (<=15s):
  1. VAD to find speech regions
  2. Energy-based sub-splitting of long regions at micro-pauses
  3. Speaker embeddings (wespeaker) per chunk
  4. Complete-linkage agglomerative clustering
  5. Per-speaker track reconstruction

Long clips (>15s):
  Full pyannote diarization pipeline.

Author: Created for Seb @ Storybook Studios
"""

import os
import torch
import torchaudio
import numpy as np
from scipy.spatial.distance import cdist
import folder_paths

# ---------------------------------------------------------------------------
# torchaudio compat shim
# ---------------------------------------------------------------------------
import torchaudio as _ta
if not hasattr(_ta, "list_audio_backends"):
    _ta.list_audio_backends = lambda: ["soundfile"]

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Model as PyannoteModel
    from pyannote.audio import Inference as PyannoteInference
    from pyannote.audio import Pipeline as PyannotePipeline
    from pyannote.audio.pipelines import VoiceActivityDetection
    PYANNOTE_AVAILABLE = True
except ImportError:
    print("[StorybookSpeechSplitter] pyannote-audio not found. "
          "Install with: pip install pyannote-audio")

SHORT_CLIP_LIMIT = 15.0


class SpeakerSeparatorNode:
    """Splits multi-speaker audio into per-speaker tracks."""

    _seg_model = None
    _emb_model = None
    _vad_pipeline = None
    _dia_pipeline = None

    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace access token.",
                }),
            },
            "optional": {
                "num_speakers": ("INT", {
                    "default": 0,
                    "min": 0, "max": 5, "step": 1,
                    "tooltip": "Exact speaker count (0=auto).",
                }),
                "min_speakers": ("INT", {
                    "default": 1,
                    "min": 1, "max": 10, "step": 1,
                    "tooltip": "Min speakers (long clips).",
                }),
                "max_speakers": ("INT", {
                    "default": 5,
                    "min": 2, "max": 10, "step": 1,
                    "tooltip": "Max speakers (long clips).",
                }),
                "speaker_similarity": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.1, "max": 1.2, "step": 0.05,
                    "tooltip": "Cosine distance threshold for "
                               "same-speaker (short clips). "
                               "Lower = stricter.",
                }),
                "fade_duration_ms": ("INT", {
                    "default": 10,
                    "min": 0, "max": 100, "step": 1,
                    "tooltip": "Fade at segment edges (ms).",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO",
                    "INT", "STRING")
    RETURN_NAMES = ("speaker_1", "speaker_2", "speaker_3",
                    "speaker_4", "speaker_5",
                    "num_speakers", "info")
    FUNCTION = "process"
    CATEGORY = "audio/speech"

    # ------------------------------------------------------------------
    # Model loaders (lazy, cached across runs)
    # ------------------------------------------------------------------
    def _get_seg_model(self, hf_token):
        if SpeakerSeparatorNode._seg_model is None:
            print("[StorybookSpeechSplitter] Loading "
                  "segmentation-3.0 ...")
            try:
                m = PyannoteModel.from_pretrained(
                    "pyannote/segmentation-3.0", token=hf_token)
            except TypeError:
                m = PyannoteModel.from_pretrained(
                    "pyannote/segmentation-3.0",
                    use_auth_token=hf_token)
            m.eval().to(self.device)
            SpeakerSeparatorNode._seg_model = m
            print("[StorybookSpeechSplitter] segmentation ready")
        return SpeakerSeparatorNode._seg_model

    def _get_vad(self, hf_token):
        if SpeakerSeparatorNode._vad_pipeline is None:
            seg = self._get_seg_model(hf_token)
            vad = VoiceActivityDetection(segmentation=seg)
            vad.instantiate({
                "min_duration_on": 0.1,
                "min_duration_off": 0.05,
            })
            SpeakerSeparatorNode._vad_pipeline = vad
            print("[StorybookSpeechSplitter] VAD ready")
        return SpeakerSeparatorNode._vad_pipeline

    def _get_emb_inference(self, hf_token):
        if SpeakerSeparatorNode._emb_model is None:
            print("[StorybookSpeechSplitter] Loading "
                  "wespeaker embeddings ...")
            try:
                m = PyannoteModel.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM",
                    token=hf_token)
            except TypeError:
                m = PyannoteModel.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM",
                    use_auth_token=hf_token)
            inf = PyannoteInference(m, window="whole")
            inf.to(torch.device(self.device))
            SpeakerSeparatorNode._emb_model = inf
            print("[StorybookSpeechSplitter] embeddings ready")
        return SpeakerSeparatorNode._emb_model

    def _get_dia_pipeline(self, hf_token):
        if SpeakerSeparatorNode._dia_pipeline is None:
            print("[StorybookSpeechSplitter] Loading "
                  "diarization pipeline ...")
            try:
                p = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token)
            except TypeError:
                p = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token)
            p.to(torch.device(self.device))
            SpeakerSeparatorNode._dia_pipeline = p
            print("[StorybookSpeechSplitter] diarization ready")
        return SpeakerSeparatorNode._dia_pipeline

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mono(waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform

    @staticmethod
    def _make_audio(wav, sr):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return {"waveform": wav.unsqueeze(0).cpu(),
                "sample_rate": sr}

    @staticmethod
    def _make_silent(sr, n):
        return {"waveform": torch.zeros(1, 1, n),
                "sample_rate": sr}

    @staticmethod
    def _apply_fade(seg_np, sr, fade_ms):
        if fade_ms <= 0 or len(seg_np) < 2:
            return seg_np
        n = min(int(sr * fade_ms / 1000), len(seg_np) // 2)
        if n < 1:
            return seg_np
        out = seg_np.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
        return out

    # ------------------------------------------------------------------
    # Energy-based sub-splitting
    # ------------------------------------------------------------------
    def _energy_subsplit(self, waveform_mono, sr, seg_start,
                         seg_end, min_chunk_s=0.5,
                         window_ms=30):
        """Split a speech region at its deepest energy dip.

        Works recursively: caller can re-split returned chunks.
        """
        from scipy.ndimage import uniform_filter1d

        s_samp = int(seg_start * sr)
        e_samp = int(seg_end * sr)
        chunk = waveform_mono[0, s_samp:e_samp].cpu().numpy()
        dur = seg_end - seg_start

        if dur < min_chunk_s * 2:
            return [(seg_start, seg_end)]

        # Short-time energy
        win = max(1, int(sr * window_ms / 1000))
        hop = max(1, win // 2)

        energy = []
        positions = []
        for i in range(0, len(chunk) - win, hop):
            frame = chunk[i:i + win]
            energy.append(np.sum(frame ** 2))
            positions.append(i + win // 2)

        if len(energy) < 5:
            return [(seg_start, seg_end)]

        energy = np.array(energy, dtype=np.float64)
        positions = np.array(positions)

        # Smooth
        smooth = uniform_filter1d(energy, size=5)

        # Search middle 80%
        margin = max(1, int(len(smooth) * 0.1))
        search = smooth[margin:-margin]
        search_pos = positions[margin:-margin]

        if len(search) < 3:
            return [(seg_start, seg_end)]

        # Find local minima
        dips = []
        for i in range(1, len(search) - 1):
            if search[i] <= search[i - 1] and \
               search[i] <= search[i + 1]:
                t = seg_start + search_pos[i] / sr
                if (t - seg_start >= min_chunk_s and
                        seg_end - t >= min_chunk_s):
                    dips.append((search[i], t))

        if not dips:
            return [(seg_start, seg_end)]

        # Pick deepest dip
        dips.sort(key=lambda x: x[0])
        best_t = dips[0][1]
        print(f"    energy split at {best_t:.2f}s "
              f"(energy={dips[0][0]:.6f})")

        return [(seg_start, best_t), (best_t, seg_end)]

    # ------------------------------------------------------------------
    # Short clip: VAD + energy split + embeddings + clustering
    # ------------------------------------------------------------------
    def _run_short_clip(self, waveform_mono, sr, hf_token,
                        similarity_thresh, num_speakers,
                        fade_ms):
        n_samples = waveform_mono.shape[-1]
        duration = n_samples / sr

        # --- Resample to 16 kHz ---
        if sr != 16000:
            wav16 = torchaudio.transforms.Resample(
                sr, 16000)(waveform_mono)
        else:
            wav16 = waveform_mono

        # --- VAD ---
        # Ensure fp32 (SAM3 may leave GPU in bf16 mode)
        wav16 = wav16.float()
        
        vad = self._get_vad(hf_token)
        speech_regions = vad(
            {"waveform": wav16, "sample_rate": 16000})

        vad_segs = [(s.start, s.end)
                    for s in speech_regions.get_timeline()]
        print(f"[StorybookSpeechSplitter] VAD: "
              f"{len(vad_segs)} region(s)")
        for i, (s, e) in enumerate(vad_segs):
            print(f"  region {i}: {s:.2f}s - {e:.2f}s "
                  f"({e - s:.2f}s)")

        if not vad_segs:
            return ([self._make_silent(sr, n_samples)],
                    0, "No speech detected")

        # --- Energy sub-split long regions ---
        chunks = []
        for vs, ve in vad_segs:
            if ve - vs > 1.5:
                subs = self._energy_subsplit(
                    waveform_mono, sr, vs, ve,
                    min_chunk_s=0.4)
                # One more level for very long regions
                final = []
                for ss, se in subs:
                    if se - ss > 2.5:
                        final.extend(self._energy_subsplit(
                            waveform_mono, sr, ss, se,
                            min_chunk_s=0.4))
                    else:
                        final.append((ss, se))
                chunks.extend(final)
            else:
                chunks.append((vs, ve))

        chunks.sort(key=lambda x: x[0])
        print(f"[StorybookSpeechSplitter] Chunks: {len(chunks)}")
        for i, (s, e) in enumerate(chunks):
            print(f"  chunk {i}: {s:.2f}s - {e:.2f}s "
                  f"({e - s:.2f}s)")

        if len(chunks) < 2:
            wav_np = waveform_mono[0].cpu().numpy()
            return ([self._make_audio(
                        torch.from_numpy(wav_np), sr)],
                    1, "Single region, no split possible")

        # --- Extract embeddings ---
        emb_inf = self._get_emb_inference(hf_token)
        embeddings = []
        valid = []

        for i, (s, e) in enumerate(chunks):
            if e - s < 0.3:
                print(f"  chunk {i}: skipped (too short)")
                continue
            s_samp = int(s * sr)
            e_samp = int(e * sr)
            chunk_wav = waveform_mono[0, s_samp:e_samp]

            if sr != 16000:
                c16 = torchaudio.transforms.Resample(
                    sr, 16000)(chunk_wav.unsqueeze(0))
            else:
                c16 = chunk_wav.unsqueeze(0)
            # c16: (1, samples) = (channel, time)

            try:
                emb = emb_inf(
                    {"waveform": c16, "sample_rate": 16000})
                embeddings.append(emb)
                valid.append((s, e, i))
            except Exception as ex:
                print(f"  chunk {i}: embedding error: {ex}")

        if len(embeddings) < 2:
            wav_np = waveform_mono[0].cpu().numpy()
            return ([self._make_audio(
                        torch.from_numpy(wav_np), sr)],
                    1, f"Not enough embeddings ({len(embeddings)})")

        # --- Cluster with complete linkage ---
        emb_matrix = np.vstack(embeddings)
        dist = cdist(emb_matrix, emb_matrix, metric="cosine")
        n = len(embeddings)

        print(f"[StorybookSpeechSplitter] Distances ({n}):")
        for i in range(n):
            ds = " ".join(f"{d:.3f}" for d in dist[i])
            s_t, e_t, ci = valid[i]
            print(f"  c{ci}({s_t:.2f}-{e_t:.2f}): [{ds}]")

        labels = list(range(n))

        def complete_dist(ci, cj):
            """Max pairwise distance between clusters."""
            mi = [x for x in range(n) if labels[x] == ci]
            mj = [x for x in range(n) if labels[x] == cj]
            return max(dist[a, b] for a in mi for b in mj)

        if num_speakers > 0:
            while len(set(labels)) > num_speakers:
                best_d, best_i, best_j = float('inf'), -1, -1
                ul = sorted(set(labels))
                for ii in range(len(ul)):
                    for jj in range(ii + 1, len(ul)):
                        d = complete_dist(ul[ii], ul[jj])
                        if d < best_d:
                            best_d, best_i, best_j = \
                                d, ul[ii], ul[jj]
                print(f"  merge {best_i}+{best_j} "
                      f"(dist={best_d:.3f})")
                labels = [best_i if l == best_j else l
                          for l in labels]
        else:
            changed = True
            while changed:
                changed = False
                best_d, best_i, best_j = float('inf'), -1, -1
                ul = sorted(set(labels))
                if len(ul) <= 1:
                    break
                for ii in range(len(ul)):
                    for jj in range(ii + 1, len(ul)):
                        d = complete_dist(ul[ii], ul[jj])
                        if d < best_d:
                            best_d, best_i, best_j = \
                                d, ul[ii], ul[jj]
                if best_d < similarity_thresh:
                    print(f"  merge {best_i}+{best_j} "
                          f"(dist={best_d:.3f})")
                    labels = [best_i if l == best_j else l
                              for l in labels]
                    changed = True

        # Remap to 0,1,2...
        ul = sorted(set(labels))
        lmap = {l: idx for idx, l in enumerate(ul)}
        labels = [lmap[l] for l in labels]
        n_spk = len(ul)

        desc = ", ".join(
            f"c{valid[i][2]}=spk{labels[i]+1}"
            for i in range(n))
        print(f"[StorybookSpeechSplitter] {n_spk} speaker(s): "
              f"{desc}")

        # --- Build per-speaker tracks ---
        wav_np = waveform_mono[0].cpu().numpy()
        results = []
        info_parts = [f"Short-clip | {n_spk} speaker(s) "
                      f"in {duration:.1f}s"]

        for spk in range(n_spk):
            out = np.zeros(n_samples, dtype=np.float32)
            total_t = 0
            segs = 0
            for idx, (s, e, ci) in enumerate(valid):
                if labels[idx] == spk:
                    ss = max(0, min(int(s * sr), n_samples))
                    es = max(0, min(int(e * sr), n_samples))
                    if es > ss:
                        seg = wav_np[ss:es].copy()
                        seg = self._apply_fade(
                            seg, sr, fade_ms)
                        out[ss:es] = seg
                        total_t += e - s
                        segs += 1
            info_parts.append(
                f"  Speaker {spk+1}: {segs} seg(s), "
                f"{total_t:.2f}s")
            results.append(self._make_audio(
                torch.from_numpy(out), sr))

        return results, n_spk, "\n".join(info_parts)

    # ------------------------------------------------------------------
    # Long clip: full diarization pipeline
    # ------------------------------------------------------------------
    def _run_diarization(self, waveform_mono, sr, hf_token,
                         min_spk, max_spk, num_speakers,
                         fade_ms):
        pipeline = self._get_dia_pipeline(hf_token)

        if sr != 16000:
            wav16 = torchaudio.transforms.Resample(
                sr, 16000)(waveform_mono)
        else:
            wav16 = waveform_mono

        kwargs = {}
        if num_speakers > 0:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["min_speakers"] = min_spk
            kwargs["max_speakers"] = max_spk

        n_samples = waveform_mono.shape[-1]
        duration = n_samples / sr

        print(f"[StorybookSpeechSplitter] Diarizing "
              f"({duration:.1f}s) ...")
        diarization = pipeline(
            {"waveform": wav16, "sample_rate": 16000},
            **kwargs)

        segs = {}
        dia = diarization
        if hasattr(diarization, 'speaker_diarization'):
            dia = diarization.speaker_diarization
            if isinstance(dia, dict):
                dia = next(iter(dia.values()))
        if hasattr(dia, 'itertracks'):
            for turn, _, spk in dia.itertracks(
                    yield_label=True):
                segs.setdefault(spk, []).append(
                    (turn.start, turn.end))

        n_spk = len(segs)
        results = []
        info_parts = [f"Diarization | {n_spk} speaker(s) "
                      f"in {duration:.1f}s"]

        wav_np = waveform_mono[0].cpu().numpy()
        for i, (spk, seg_list) in enumerate(
                sorted(segs.items())):
            total_t = sum(e - s for s, e in seg_list)
            info_parts.append(
                f"  Speaker {i+1} ({spk}): "
                f"{len(seg_list)} seg(s), {total_t:.2f}s")
            out = np.zeros(n_samples, dtype=np.float32)
            for s_t, e_t in seg_list:
                ss = max(0, min(int(s_t * sr), n_samples))
                es = max(0, min(int(e_t * sr), n_samples))
                if es > ss:
                    seg = wav_np[ss:es].copy()
                    seg = self._apply_fade(seg, sr, fade_ms)
                    out[ss:es] = seg
            results.append(self._make_audio(
                torch.from_numpy(out), sr))

        return results, n_spk, "\n".join(info_parts)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def process(
        self,
        audio,
        hf_token="",
        num_speakers=0,
        min_speakers=1,
        max_speakers=5,
        speaker_similarity=0.65,
        fade_duration_ms=10,
    ):
        if not PYANNOTE_AVAILABLE:
            raise RuntimeError(
                "pyannote-audio not installed. "
                "pip install pyannote-audio")
        if not hf_token:
            raise ValueError("HuggingFace token required.")

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        waveform_mono = self._to_mono(waveform.squeeze(0))
        n_samples = waveform_mono.shape[-1]
        duration = n_samples / sr

        if duration <= SHORT_CLIP_LIMIT:
            print(f"[StorybookSpeechSplitter] Short clip "
                  f"({duration:.1f}s @ {sr}Hz)")
            results, n_spk, info = self._run_short_clip(
                waveform_mono, sr, hf_token,
                speaker_similarity, num_speakers,
                fade_duration_ms)
        else:
            print(f"[StorybookSpeechSplitter] Long clip "
                  f"({duration:.1f}s @ {sr}Hz)")
            results, n_spk, info = self._run_diarization(
                waveform_mono, sr, hf_token,
                min_speakers, max_speakers, num_speakers,
                fade_duration_ms)

        while len(results) < 5:
            results.append(self._make_silent(sr, n_samples))

        print(f"[StorybookSpeechSplitter] Done\n{info}")
        return (*results[:5], n_spk, info)


# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SpeakerSeparator": SpeakerSeparatorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerSeparator": "Storybook Speech Splitter",
}
