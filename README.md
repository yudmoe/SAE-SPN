# AESPN (NYU Dataset)

The NYU training and testing code for the AESPN portion has been officially released. Plans for subsequent releases are currently being developed.

## 🛠️ Requirements

Ensure you have the following dependencies installed before running the code:
* `pytorch_lightning`
* *(Please add other necessary packages here, e.g., torch, numpy)*

## 📥 Checkpoints

The pre-trained model weights can be downloaded via Baidu Netdisk:
* **File:** `epoch=82-RMSE=0.0891.ckpt`
* **Link:** [Baidu Netdisk (百度网盘)](https://pan.baidu.com/s/1ilv5VDz2IaVQnGoduxg8oQ)
* **Access Code (提取码):** `tu41`

## 🚀 Usage

The main execution file for this project is `lit_NYU_main_customLoss_dataset_prefill.py`.

### Step-by-Step Instructions

| Phase | Action Required | Target File | Description |
|---|---|---|---|
| **1. Data Preparation** | Modify `dir_data` and `split_json` | `datasetsettings_NYU.py` | Configure these parameters to correctly locate and prepare your NYU dataset. |
| **2. Testing** | Set `pretrain_weight = True` <br> Update checkpoint path | `settings_NYU.py` (Line 54) | Specify the path to the downloaded checkpoint to evaluate the model. |
| **3. Training** | Set `pretrain_weight = False` | `settings_NYU.py` | Run `lit_NYU_main_customLoss_dataset_prefill.py` to start training your model from scratch. |
