model_name =    "AE_SPN"

seed = 3407
n_device = 1
# dataset
dataset_py =   'nyu_prefill'
data_name = "NYU"
dir_data = '/data/zzy/NYU_DepthV2_HDF5/nyudepthv2'
split_json = "/home/zzy/code/my_NLSPN/data_json/nyu.json"
augment = True
num_sample = 500
patch_height = 228
patch_width = 304
#LC_MODE
sample_mode = "center"
coverage = 0.05

# base network
norm_depth = [0.2, 10.0]
basemodel = "v1"
resnet = "res34"
sto_depth = True
pretrain_weight = None
val_output = True
resume_weight = None
# SPN
spn_enable = True
spn_module = 'SAESPN_model' 
prop_kernel = 5
prop_time = 24
# loss
loss_name =  "sloss_onlylast"
downLR1 = 40
downLR2 = 60
w_1 = 1.0
w_2 = 1.0
# dataloader
n_thread = 8
n_batch = 3
# met
eval_range = None
# optimizer
learning_rates = 5e-4
w_weight_decay = 0
warm_up_epochs = 1
epochs = 100
step = 40
LR_down_gamma = 0.5
#test
test_only = True
if test_only:
    n_device = 1
    top_crop = 0
    pretrain_weight = "/data2/zzy/lightning_experiments/version_2735_AESPN_NYU_89/checkpoints/epoch=82-RMSE=0.0891.ckpt"
    resume_weight = "/data2/zzy/lightning_experiments/version_2735_AESPN_NYU_89/checkpoints/epoch=82-RMSE=0.0891.ckpt"