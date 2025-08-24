from glob import glob
import os
import json

if __name__ == "__main__":
    image_paths = glob("data/*")
    data = []
    
    for image_path in image_paths:
        caption = "black wavy french bob vibes from 1920s"
        image_file_name = os.path.basename(image_path)
        line = {
            "file_name": image_file_name,
            "text": caption,
        }
        data.append(line)

    with open("data/metadata.jsonl", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")