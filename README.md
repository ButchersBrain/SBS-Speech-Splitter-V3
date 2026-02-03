# ComfyUI Speaker Separator Node

A custom ComfyUI node that separates dialogue from multiple speakers in an audio clip. Uses **pyannote-audio** for state-of-the-art speaker diarization.

## What It Does

**Input:** A 5-second clip with 3 people talking  
**Output:** 3 audio files (5 seconds each), one per speaker, with other speakers silenced

The node:
1. Analyzes the audio to detect who speaks when (diarization)
2. Creates separate audio tracks for each detected speaker
3. Silences all other speakers in each track
4. Applies smooth fade transitions to avoid audio clicks

## Installation

### 1. Clone to Custom Nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_REPO/comfyui_speaker_separator
# Or copy the folder directly
```

### 2. Install Dependencies

```bash
cd comfyui_speaker_separator
pip install -r requirements.txt
```

### 3. Get HuggingFace Access (Required!)

Pyannote models require authentication:

1. Create a HuggingFace account at https://huggingface.co
2. Get an access token at https://huggingface.co/settings/tokens
3. **Accept the model terms** at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
4. Paste your token in the node's `hf_token` field

## Node Inputs

| Input | Type | Description |
|-------|------|-------------|
| `audio` | AUDIO | ComfyUI audio input (waveform + sample_rate) |
| `hf_token` | STRING | Your HuggingFace access token |
| `max_speakers` | INT | Maximum speakers to detect (2-10, default: 5) |
| `min_speakers` | INT | Minimum speakers expected (default: 1) |
| `fade_duration_ms` | INT | Crossfade at boundaries to avoid clicks (0-100ms) |

## Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| `speaker_1` | AUDIO | Audio with only speaker 1, others silenced |
| `speaker_2` | AUDIO | Audio with only speaker 2, others silenced |
| `speaker_3` | AUDIO | Audio with only speaker 3, others silenced |
| `speaker_4` | AUDIO | Audio with only speaker 4, others silenced |
| `speaker_5` | AUDIO | Audio with only speaker 5, others silenced |
| `num_speakers` | INT | Number of speakers detected |
| `diarization_info` | STRING | Detailed info about segments per speaker |

Unused speaker outputs will be silent audio tracks of the same length.

## Example Workflow

```
[Load Audio] → [Speaker Separator] → [Save Audio] (x5)
                     ↓
              [Preview Audio]
```

### Typical Use Case (Your Scenario)

```
Input: 5-second clip with Person A, Person B, Person C talking

Output:
  speaker_1.wav → 5 seconds, only Person A's voice, silence elsewhere
  speaker_2.wav → 5 seconds, only Person B's voice, silence elsewhere  
  speaker_3.wav → 5 seconds, only Person C's voice, silence elsewhere
  speaker_4.wav → 5 seconds, silent (no 4th speaker)
  speaker_5.wav → 5 seconds, silent (no 5th speaker)
  num_speakers → 3
```

## Technical Notes

### How It Works

1. **Pyannote Diarization**: The node uses `pyannote/speaker-diarization-3.1`, which:
   - Segments the audio into speech regions
   - Clusters segments by speaker identity
   - Outputs timestamp ranges per speaker

2. **Audio Separation**: For each detected speaker:
   - Creates a silent track of the full duration
   - Copies audio only from that speaker's segments
   - Applies fade-in/out at segment boundaries

### Limitations

- **No true source separation**: If speakers overlap/talk over each other, both voices appear in both tracks during that overlap. The node silences based on "primary speaker" timestamps.
- **Requires clear speech**: Works best when speakers take turns. Heavy overlapping speech reduces accuracy.
- **GPU recommended**: Pyannote runs much faster on CUDA GPU.

### Alternative Approaches

If you need **true source separation** (isolating overlapping voices), consider:
- **SepFormer** (SpeechBrain) - Good for 2-3 overlapping speakers
- **Conv-TasNet** - Real-time capable separation
- **Whisper + diarization** - For transcription with speaker labels

These could be added as alternative modes in future versions.

## Troubleshooting

### "pyannote-audio not installed"
```bash
pip install pyannote-audio
```

### "HuggingFace token required"
1. Get token: https://huggingface.co/settings/tokens
2. Accept model terms (links above)
3. Paste token in node

### "Model not found / 401 error"
You haven't accepted the model license terms. Visit both pyannote model pages and click "Agree and access repository".

### Slow processing
- First run downloads ~1GB of models
- Use GPU if available (`torch.cuda.is_available()`)
- Longer audio = longer processing

## License

MIT License - Use freely in your projects.

## Credits

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Hervé Bredin et al.
- Created for VFX/AI film production workflows
