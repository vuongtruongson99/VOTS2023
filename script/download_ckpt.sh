pip install gdown

# download mae-vit
wget -P ./pretrained_models https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# download procontext-tracking-net model
gdown --id '11Xh5puRMHAN72wVKtwaRRgNdSFCUsY1Y' --output ./output/checkpoints/train/procontext/procontext/ProContEXT_ep0300.pth.tar

# download procontext-got-10k model
gdown --id '1K1mf2CqoYuTevBaPJDf3trQAS3hbzMFp' --output ./output/checkpoints/train/procontext/procontext_got10k/ProContEXT_ep0100.pth.tar

# download aot-ckpt 
gdown --id '1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq' --output ./ckpt/SwinB_DeAOTL_PRE_YTB_DAV.pth

# download sam-ckpt
# wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# download grounding-dino ckpt
wget -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth