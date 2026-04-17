import os
import shutil
import pandas as pd
import random

image_source = "FETAL_PLANES_ZENODO/Images"
csv_file = "FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv"

dest = "plane_classifier/data"

data = pd.read_csv(csv_file, sep=";")

print("Columns:",data.columns)

for folder in ["train","val"]:
    for cls in ["head","abdomen","other"]:
        os.makedirs(os.path.join(dest,folder,cls),exist_ok=True)

def map_class(plane):

    if "brain" in plane.lower():
        return "head"

    elif "abdomen" in plane.lower():
        return "abdomen"

    else:
        return "other"


groups={
"head":[],
"abdomen":[],
"other":[]
}

for i,row in data.iterrows():

    img=row["Image_name"] + ".png"

    plane=row["Plane"]

    cls=map_class(plane)

    groups[cls].append(img)


for cls in groups:

    images=groups[cls]

    random.shuffle(images)

    split=int(0.8*len(images))

    train=images[:split]

    val=images[split:]

    for img in train:

        src=os.path.join(image_source,img)

        if os.path.exists(src):

            shutil.copy(
                src,
                os.path.join(dest,"train",cls,img)
            )

    for img in val:

        src=os.path.join(image_source,img)

        if os.path.exists(src):

            shutil.copy(
                src,
                os.path.join(dest,"val",cls,img)
            )

    print(cls,"done")

print("Dataset prepared successfully")