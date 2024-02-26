#!/bin/bash
pip install gdown
# download the dataset
PathToDataset="./datasets/MMCoQA/"
mkdir -p $Path_To_Dataset
gdown 1QI_lQfM4xvDyNs2gCiY-eIR-8pGh9KJ2 -O $Path_To_Dataset


# download the features
PathToFeatures=./stored_features/image/clip_features/

mkdir -p $PathToFeatures
if [ "$(ls -A $PathToFeatures)" ]; then
    echo "image features file exists!"
    # Perform some action here
else
    echo "downloading image features"
    gdown 15RzG-KL7EwmerV2nTzJKSprODki_3rbf -O $PathToFeatures
fi


PathToFeatures=stored_features/table/ada_features/
mkdir -p $PathToFeatures
if [ "$(ls -A $PathToFeatures)" ]; then
    echo "table features file exists!"
    # Perform some action here
else
    echo "downloading table features"
    gdown 1MrTzFuvpczPthdc_RwkahAYMkk6OeGw4 -O $PathToFeatures
    gdown 1VbDAr_35G7d_eucciL3RM-jAF5yHzydF -O $PathToFeatures
fi

