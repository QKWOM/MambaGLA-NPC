import os
from email.policy import strict

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from myDataset_test import MyDataset_test
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn.functional as F
from monai.metrics import compute_hausdorff_distance

os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
# from model_segmamba.segmamba_CNN import SegMamba
# from model_segmamba.experiment.raw_mamba import SegMamba
# from model_segmamba.experiment.CNN_Attention_mamba import SegMamba
# from model_segmamba.experiment.Attention_mamba import SegMamba
# from model_segmamba.experiment.CNN_Attention_Tmamba import SegMamba
# from model_segmamba.experiment.CNN_Attention_Tmamba2 import SegMamba
# from model_segmamba.experiment.just_local_scan import SegMamba
# from model_segmamba.experiment.just_local_scan import SegMamba
# from model_segmamba.experiment.just_local_scan import SegMamba

# from model_segmamba.experiment.Apply_UXNet_encoder import SegMamba
# from model_segmamba.experiment.Apply_SwinTransformer_encoder import SegMamba

# from model_segmamba.Final_experiment.just_vertical_scan import SegMamba
# from model_segmamba.Final_experiment.just_horizontal_scan import SegMamba
# from model_segmamba.Final_experiment.local_vertical_scan import SegMamba
# from model_segmamba.Final_experiment.local_horizontal_scan import SegMamba
# from model_segmamba.Final_experiment.vertical_horizontal_scan import SegMamba
# from model_segmamba.Final_experiment.local_vertical_horizontal_scan import SegMamba
from model_segmamba.Final_experiment.local_vertical_scan_attention import SegMamba
# from model_segmamba.Final_experiment.local_scan import SegMamba
# from model_segmamba.Final_experiment.without_CNN_scan import SegMamba
# from model_segmamba.Final_experiment.without_GLSSM_scan import SegMamba
# from model_segmamba.Final_experiment.without_BiAtt_scan import SegMamba
# from model_segmamba.Final_experiment.without_GCSAtt_scan import SegMamba

def get_dice_score(prev_masks, gt3D):
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    pred_masks = (prev_masks > 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
    return (sum(dice_list) / len(dice_list)).item()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        input = nn.Sigmoid()(input)
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# image_path = r"/opt/data/private/workspace/hzh/all_data/nose_cancel_add_test/image/"
# label_path = r"/opt/data/private/workspace/hzh/all_data/nose_cancel_add_test/label/"
image_path = r"/opt/data/private/workspace/hzh/all_data/nose_cancel_all_test_new/image/"
label_path = r"/opt/data/private/workspace/hzh/all_data/nose_cancel_all_test_new/label/"
# image_path = r"/opt/data/private/workspace/hzh/all_data/hospital2_test_new/image"
# label_path = r"/opt/data/private/workspace/hzh/all_data/hospital2_test_new/label"
# image_path = r"/opt/data/private/workspace/hzh/all_data/hospital1_test_new/image"
# label_path = r"/opt/data/private/workspace/hzh/all_data/hospital1_test_new/label"

save_path = "./best2.pth"
resume = True

mode_type = ['grid_and_channel','concatenation','enhanced_grid_and_channel']
net = SegMamba(out_chans=1,mode=mode_type[0]).to(device)
# net = SegMamba(out_chans=1).to(device)
# net = None
if resume :
    # state_dict = torch.load("./best_ori_0.7600.pth")
    # net.load_state_dict(state_dict)
    state_dict = torch.load("./test121_0.8047.pth", map_location=device)
    filter_model_state = {k: v for k, v in state_dict['model_state_dict'].items()
                   if 'total_ops' not in k and 'total_params' not in k}
    net_state_dict = net.state_dict()
    pretrained_dict = {
        k: v for k, v in filter_model_state.items()
        if k in net_state_dict and v.shape == net_state_dict[k].shape
    }
    net.load_state_dict(pretrained_dict, strict=False)
train_dataset = MyDataset_test(image_path=image_path, label_path=label_path)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


Epoch = 80
epoch_best_loss = 100

net.eval()
dice_list = []
recall_list = []
accuracy_list = []
precision_list = []
IOU_list = []
HD95_list = []
loss_fn = DiceLoss()
maxn = 0
minn = 1000
min_tensor = None
max_tensor = None
min_info = None
max_info = None
min_path = None
max_path = None
max_image = None
min_image = None
max_lis = []
min_lis = []

def save_nii2(in_arr, out_path, meta_info):
    in_arr = in_arr.detach().cpu().numpy().astype(np.float32)
    in_arr = in_arr.squeeze(0)
    ori_arr = np.transpose(in_arr, [1,2,3,0])
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)

def save_nii(in_arr, out_path, meta_info):
    in_arr = in_arr.detach().cpu().numpy().astype(np.float32)
    in_arr = in_arr.squeeze(0)
    ori_arr = np.transpose(in_arr, [1,2,3,0])
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)

with torch.no_grad():
    epoch_loss = 0
    dice_all = 0
    for index, (img, mask, info) in enumerate(tqdm(train_loader, total=len(train_loader))):
        img, mask = img.to(device), mask.to(device)
        # print(img.shape)
        # pred,mamba,cnn,enc,g_enc = net(img)
        pred = net(img)
        ##########
        # count = 0
        # if count==0:
        #     for x,y,e,ge in zip(mamba,cnn,enc,g_enc):
        #         count += 1
                # if count==1:
                #     save_nii2(img, f"./middle_out/ori_image.nii", info)
                # save_nii2(x, f"./middle_out/mmaba{count}.nii", info)
                # save_nii2(y, f"./middle_out/cnn{count}.nii", info)
                # save_nii2(e, f"./middle_out/enc{count}.nii", info)
                # save_nii2(ge, f"./middle_out/g_enc{count}.nii", info)
        ##########
        dice = get_dice_score(pred, mask)
        dice_all += dice

        prev_mask = (pred>0.5)
        
        save_mask = pred

        prev_mask = prev_mask.cpu()
        mask = mask.cpu()
        smooth = 1e-8
        true_positive = np.logical_and(mask,prev_mask)
        false_negative = np.logical_and(mask,np.logical_not(prev_mask))
        true_negative = np.logical_and(np.logical_not(prev_mask),np.logical_not(mask))
        false_positive = np.logical_and(np.logical_not(mask),prev_mask)

        #HD95计算
        hd95_value = compute_hausdorff_distance(prev_mask, mask, distance_metric='euclidean', percentile=95)
        HD95_list.append(hd95_value[0][0])

        # precision计算
        precision = true_positive.sum()/(true_positive.sum()+false_positive.sum()+smooth)
        # print(precision)
        precision_list.append(precision)

        # dice计算
        dice = (2*true_positive.sum()+smooth)/(2*true_positive.sum()+false_negative.sum()+false_positive.sum()+smooth)
        dice_list.append(dice)

        if dice>0.9:
            max_lis.append(info['path'])
        if dice<0.5:
            min_lis.append(info['path'])
        if dice>maxn:
            maxn=dice
            max_tensor=save_mask
            max_info = info
            max_path = info['path']
            max_image = img
            max_label = mask

        if dice<minn:
            minn=dice
            min_tensor=save_mask
            min_info = info
            min_path = info['path']
            min_image = img
       
        # IOU计算
        iou = true_positive.sum()/(true_positive.sum()+false_negative.sum()+false_positive.sum()+smooth)
        IOU_list.append(iou)

        # accuracy计算
        accuracy = (true_positive.sum()+true_negative.sum())/(true_positive.sum()+true_negative.sum()+false_positive.sum()+false_negative.sum())
        accuracy_list.append(accuracy)
        # recall计算
        recall = true_positive.sum()/(true_positive.sum()+false_negative.sum())
        recall_list.append(recall)

    dice_all /= len(train_loader)
    print("Dice:", dice_all)
print('IOU:',np.mean(np.array(IOU_list)))
print('accuracy:',np.mean(np.array(accuracy_list)))
print('precision:',np.mean(np.array(precision_list)))
print('recall:',np.mean(np.array(recall_list)))
print('dice:',np.mean(np.array(dice_list)))
print('HD95:',np.mean(np.array(HD95_list)))
print('max_dice:',maxn)
print('max_path:',max_path)
print('min_dice:',minn)
print('min_path:',min_path)
print(max_lis)
print(min_lis)
min_tensor = min_tensor.detach().cpu().numpy().astype(np.float32)
max_tensor = max_tensor.detach().cpu().numpy().astype(np.float32)
max_image = max_image.detach().cpu().numpy().astype(np.float32)
min_image = min_image.detach().cpu().numpy().astype(np.float32)


# save_nii(min_tensor, "min.nii", min_info)
# save_nii(max_tensor, "max.nii", max_info)
# save_nii(max_image, "max_image.nii", max_info)
# save_nii(min_image, "min_image.nii", min_info)
#
# save_nii(max_label, "max_label.nii", max_info)



