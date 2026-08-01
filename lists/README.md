# Lists

Training lists use one line per utterance:

```text
<speaker_id> <relative_audio_path>
```

Example:

```text
id10001 id10001/1zcIwhmdeo4/00001.wav
```

Trial lists use the VoxCeleb format:

```text
<label> <enroll_relative_audio_path> <test_relative_audio_path>
```

`label` is `1` for target trials and `0` for non-target trials. Paths are resolved under the configured dataset root.
