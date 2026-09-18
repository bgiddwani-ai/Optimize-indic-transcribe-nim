import pathlib
import soundfile as sf

def dir2manifest(file_path:pathlib.Path, file_key='audio_filepath', duration_key='duration', max_entry=None):
    manifest=[]
    count=0
    def file2json(filename):
        try:
            duration=sf.info(filename).duration
        except Exception as exp:
            raise Exception(f"Could not read {filename}. Likely not an audio file")
        else:
            manifest.append({file_key:str(filename),duration_key:duration})

    if file_path.is_file():
        file2json(file_path)
    elif file_path.is_dir():
        for f in file_path.glob('**/*.wav'):
            if f.is_file():
                print(f)
                file2json(f)
            count+=1
            if max_entry is not None:
                if count >= max_entry:
                    break


    return manifest







