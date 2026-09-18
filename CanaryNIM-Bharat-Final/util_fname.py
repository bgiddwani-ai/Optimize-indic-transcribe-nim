import json

with open('/home/CanaryNIM-Bharat-Final/dataset/hi_manifest_cleaned.json', "r") as f:
    data = []
    for sample in f:
        data.append(json.loads(sample))

new_data = [{"audio_filepath": sample['audio_filepath'].replace("/home/bodhan/valid/","/home/CanaryNIM-Bharat-Final/dataset/"),"duration": sample["duration"], "text": sample["text"]}  for sample in data]

with open('/home/CanaryNIM-Bharat-Final/dataset/hi_manifest_cleaned_final.json', "w") as g:
    for sample in new_data:
        g.write(json.dumps(sample))
        g.write("\n")

