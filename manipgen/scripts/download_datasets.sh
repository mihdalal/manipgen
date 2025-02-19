#!/bin/bash

# Download ManipGen-UniDexGrasp
if [ ! -d "assets/unidexgrasp" ]; then
    wget https://huggingface.co/datasets/minliu01/ManipGen-UniDexGrasp/resolve/main/manipgen-unidexgrasp_part_aa
    wget https://huggingface.co/datasets/minliu01/ManipGen-UniDexGrasp/resolve/main/manipgen-unidexgrasp_part_ab
    wget https://huggingface.co/datasets/minliu01/ManipGen-UniDexGrasp/resolve/main/manipgen-unidexgrasp_part_ac
    wget https://huggingface.co/datasets/minliu01/ManipGen-UniDexGrasp/resolve/main/manipgen-unidexgrasp_part_ad
    wget https://huggingface.co/datasets/minliu01/ManipGen-UniDexGrasp/resolve/main/manipgen-unidexgrasp_part_ae
    wget https://huggingface.co/datasets/minliu01/ManipGen-UniDexGrasp/resolve/main/manipgen-unidexgrasp_part_af

    cat manipgen-unidexgrasp_part_a* > manipgen-unidexgrasp.tar.gz 
    tar -xvf manipgen-unidexgrasp.tar.gz -C assets/

    rm manipgen-unidexgrasp_part_a*
    rm manipgen-unidexgrasp.tar.gz
fi

# Download ManipGen-PartNet
if [ ! -d "assets/partnet" ]; then
    wget https://huggingface.co/datasets/minliu01/ManipGen-PartNet/resolve/main/manipgen-partnet.tar.gz
    tar -xvf manipgen-partnet.tar.gz -C assets/
    rm manipgen-partnet.tar.gz
fi
