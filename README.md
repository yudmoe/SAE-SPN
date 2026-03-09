The NYU training and testing code for the AESPN portion has been released, and plans for subsequent releases are being developed.

need pytorch_lightning ( and some other pack)

Main file is the lit_NYU_main_customLoss_dataset_prefill.py

checkpoint can be downloaded by this link:通过网盘分享的文件：epoch=82-RMSE=0.0891.ckpt
链接: https://pan.baidu.com/s/1ilv5VDz2IaVQnGoduxg8oQ 提取码: tu41

Modify 【dir_data】 and 【split_json】 in datasetsettings_NYU.py to prepare your NYU dataset.
Modify the path to 【pretrain_weight】 on line 54 of settings_NYU.py and set 【pretrain_weight】 to True for testing.
Set【pretrain_weight】 to False and run  lit_NYU_main_customLoss_dataset_prefill.py to train your model.
