import json
import click

def cleanup(in_mf, out_mf):
    clean_keys=["audio_filepath", "duration", "text"]
    task_info = {"taskname": "asr",
                 "source_lang": "en",
                 "target_lang": "en",
                 "pnc": 'yes',
                 "answer": 'na'}

    with open(in_mf, 'r') as imf, open(out_mf, 'w') as omf:
        for text in imf:
            in_data=json.loads(text)
            clean_mf={k:in_data[k] for k in clean_keys}
            clean_mf.update(task_info)

            omf.write(json.dumps(clean_mf))
            omf.write("\n")



@click.command()
@click.argument('input_file')
@click.argument('output_file')
def main(input_file, output_file):
    cleanup(input_file, output_file)


if __name__ == '__main__':
    main()