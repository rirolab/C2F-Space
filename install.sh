conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
pip install -e coarse_vlm/
pip uninstall opencv-python opencv-contrib-python -y
pip cache purge
pip install opencv-contrib-python
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1%2Bcu117.html
pip install -r requirements.txt
pip install openai==1.59.3
cd sgg/Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install pytorch-lightning==2.2.5
pip install yacs ogb
pip install performer-pytorch==1.1.4 --no-deps
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install einops==0.8.1 local-attention==1.10.0 axial_positional_embedding==0.2.1
pip install supervision==0.21.0
cd ../../
pip install -e fine_refinement/