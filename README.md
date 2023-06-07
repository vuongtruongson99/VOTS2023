# ProContEXT

## ProContEXT: Exploring Progressive Context Transformer for Tracking

## TODO List
- [x] Custom data cho fit với dataloader
- [x] Chạy n tracker ứng với n object trong từng seq
- [x] Visualize kết quả xem sao
- [x] Chạy với multithread
- [x] Update toolkit lên 0.6.2
- [x] Đặt threshold cho tracker
- [x] Tích hợp với SAM
- [x] Chạy online nhiều tracker + SAM
- [x] Kết hợp output mask với SAM + DeAOT
- [ ] Đọc và tìm hiểu thêm về DeAOT

[ProContEXT](https://arxiv.org/abs/2210.15511) achieves SOTA performance on multiple benchmarks.

| Tracker     | GOT-10K (AO) | TrackingNet (AUC) |
|:-----------:|:------------:|:-----------:|
| ProContEXT | 74.6         | 84.6        |


## Quick Start

### Installation
You can refer to [OSTrack](https://github.com/botaoye/OSTrack) to install the whole environments and prepare the data.


```bash
bash script/download_ckpt.sh
bash script/install.sh
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- vot2023
            |-- animal
            |-- ants1
            |-- bag
            ...
   ```


### Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
### Train
We use models offered by [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) as our pretrained models. Put it under directory:
```
${PROJECT_ROOT}
    -- pretrained_models
      | -- mae_pretrain_vit_base.pth
```
Run the following command to train the model:
```shell
python tracking/train.py --script procontext --config procontext_got10k --save_dir ./output --mode multiple --nproc_per_node 4 # GOT-10k model
python tracking/train.py --script procontext --config procontext --save_dir ./output --mode multiple --nproc_per_node 4 # TrackingNet model
```

### Test
Mở jupyter notebook và chạy file đó

