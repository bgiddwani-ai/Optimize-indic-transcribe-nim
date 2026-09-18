import pathlib
import soundfile as sf
import json

def load_manifest(file_path: pathlib.Path, file_key='audio_filepath', duration_key='duration', text_key='text', max_entry=None):
    manifest = []

    with open(file_path, 'r') as file:
        for line in file:
            row = json.loads(line)
            duration = sf.info(row['audio_filepath']).duration
            print(row)
            manifest.append({file_key: row['audio_filepath'], duration_key: duration, text_key: row[text_key]})
            if max_entry is not None and len(manifest) == max_entry:
                break

    return manifest

