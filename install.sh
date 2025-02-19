wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar -xvf git-lfs-linux-amd64-v3.4.1.tar.gz
./git-lfs-3.4.1/install.sh
rm -rf git-lfs-linux-amd64-v3.4.1.tar.gz
rm -rf git-lfs-3.4.1
git lfs install
git lfs track "*.plugin"
./isaacgym/create_conda_env_rlgpu.sh # make sure you have mamba, this will build your conda env very fast
pip install -e isaacgym/python/
pip install -e IsaacGymEnvs
pip install -e industreallib
pip install -e rl_games
pip install -e Depth-Anything
pip install -e . # install manipgen
pip install -r requirements.txt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e robomimic
