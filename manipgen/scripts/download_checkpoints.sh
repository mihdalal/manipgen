#!/bin/bash

mkdir real_world/checkpoints/

wget https://huggingface.co/minliu01/ManipGen/resolve/main/pick.pth -O real_world/checkpoints/pick.pth
wget https://huggingface.co/minliu01/ManipGen/resolve/main/place.pth -O real_world/checkpoints/place.pth
wget https://huggingface.co/minliu01/ManipGen/resolve/main/grasp_handle.pth -O real_world/checkpoints/grasp_handle.pth
wget https://huggingface.co/minliu01/ManipGen/resolve/main/close.pth -O real_world/checkpoints/close.pth
wget https://huggingface.co/minliu01/ManipGen/resolve/main/open.pth -O real_world/checkpoints/open.pth
wget https://huggingface.co/minliu01/ManipGen/resolve/main/neural_mp.pth -O real_world/checkpoints/neural_mp.pth
