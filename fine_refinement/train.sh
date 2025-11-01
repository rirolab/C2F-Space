cd scripts/C2F-Space
python generate_c2fsuperpixels_raw.py
python prepare_c2f_pygsource.py 
cd ../..
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/GPS/c2fsuperpixels-GPS.yaml model.loss_fun focal_loss dataset.slic_compactness 10 model.residual True
