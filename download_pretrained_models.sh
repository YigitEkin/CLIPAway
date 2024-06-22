mkdir ckpts ckpts/AlphaCLIP ckpts/IPAdapter ckpts/CLIPAway
echo "Downloading Alpha-CLIP weights..."
cd ckpts/AlphaCLIP
gdown 1JfzOTvjf0tqBtKWwpBJtjYxdHi-06dbk
echo "Downloading IP-Adapter weights..."
cd ../IPAdapter
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
mkdir image_encoder && cd image_encoder
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json
cd ../../CLIPAway
echo "Downloading CLIPAway weights..."
gdown 1lFHAT2dF5GVRJLxkF1039D53gixHXaTx
cd ../../
echo "Finished downloading pretrained models."