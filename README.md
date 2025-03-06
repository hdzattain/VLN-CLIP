# VLN-CLIP


1 克隆仓库并安装依赖：

git clone https://github.com/hdzattain/VLN-CLIP

cd VLN-CLIP

pip install -r requirements.txt


2 下载RoboTHOR数据集并解压：
wget https://prior-model-weights.s3.us-east-2.amazonaws.com/robothor-challenge-2021.zip

unzip robothor-challenge-2021.zip -d data/


3 运行模型微调：
python scripts/finetune_clip.py --config configs/config.yaml

4 导出ONNX模型：
python scripts/export_onnx.py --config configs/config.yaml --checkpoint clip_vln.pth


5 构建TensorRT引擎：
bash scripts/build_trt_engine.sh


6 在Jetson Nano上部署：
python deploy/jetson_nano/deploy_jetson.py
