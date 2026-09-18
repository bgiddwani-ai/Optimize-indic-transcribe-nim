import json
from tabnanny import check

import click
import logging

def get_transcripts(manifest):
    transcripts={}
    count={}
    with open(manifest) as f:
        for line in f:
            in_data=json.loads(line)
            fname=in_data['audio_filepath'].split('/')[-1]
            if fname in transcripts:
                prev_fname=fname
                count[fname] = count.get(fname, 0) + 1
                fname=f"{fname}.{count[fname]}"

                logging.warning(f"duplicate filename {prev_fname} encountered {count[fname]} time(s), reassining to {fname}")

            transcripts[fname]=(in_data)

    return transcripts




def check_mm(riva_output, nemo_output):
    transcripts_riva = get_transcripts(riva_output)
    transcripts_nemo = get_transcripts(nemo_output)
    for fname in transcripts_nemo:
        gnd_truth=transcripts_nemo[fname]['text']
        nemo_text=transcripts_nemo[fname]['punctuated']
        riva_text=transcripts_riva[fname]['text']
        if nemo_text.strip() != riva_text.strip():
            print(f"{fname} text mismatch")
            print(f"Ground text : {gnd_truth}")
            print(f"Nemo text   : {nemo_text}")
            print(f"Riva text   : {riva_text}")

@click.command()
@click.argument("nemo_file")
@click.argument("riva_file")
def main(nemo_file, riva_file):
    check_mm(riva_file, nemo_file)

if __name__ == "__main__":
    main()




