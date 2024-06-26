file_dir = "/fs/scratch/PAS2473/zhongwei_models/fasterllava_dataset/Finetune"
import os
from tqdm import tqdm
import pdb
f2 = open("/users/PAS2473/brucewan666/Faster-LLaVA/LLaVA-Prumerge-new/playground/data/llava_v1_5_mix665k_ignore_misssing.json", "w", encoding="utf-8")
l = list()
with open("/users/PAS2473/brucewan666/Faster-LLaVA/LLaVA-Prumerge-new/playground/data/llava_v1_5_mix665k.json", "r", encoding="utf-8") as f:
    import json
    d = json.loads(f.read())
    f1 = open("/users/PAS2473/brucewan666/Faster-LLaVA/LLaVA-Prumerge-new/missing_images.txt", "w", encoding="utf-8")
    for i in tqdm(range(len(d))):
        if d[i].get("image", None) == None:
            f1.write("missing_image_id:"+str(i)+"\n")
            continue
        image_path = os.path.join(file_dir, d[i]["image"])
        if not os.path.exists(image_path):
            f1.write("missing_image:"+image_path+"\n")
            continue
        l.append(d[i])
    json.dump(l, f2)