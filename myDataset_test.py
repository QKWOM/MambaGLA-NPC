import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from torchio.data.io import sitk_to_nib

class CustomTransform(tio.Transform):
    def __init__(self):
        super().__init__()

    def apply_windowing(self, data, w_width, w_center):
        """Adjust image data based on window width and center."""
        val_min = w_center - (w_width / 2)
        val_max = w_center + (w_width / 2)
        return np.clip(data, val_min, val_max)

    def extract_tissue(self, data, label):
        """Extract tissue region based on label and compute window width and center."""
        tissue_values = data[label == 1]

        if tissue_values.size == 0:
            raise ValueError("No tissue values found in the labeled region.")

        min_val = np.min(tissue_values)
        max_val = np.max(tissue_values)

        w_center = (np.float64(max_val) + np.float64(min_val)) / 2
        w_width = max_val - min_val

        return w_width, w_center, min_val, max_val

    def apply_transform(self, subject):
        # Assuming input data is in subject
        image_data = subject['image'][tio.DATA].numpy()  # Convert to NumPy array
        label_data = subject['label'][tio.DATA].numpy()  # Convert to NumPy array

        # Compute window width and center
        w_width, w_center, min_val, max_val = self.extract_tissue(image_data, label_data)

        # Adjust image
        adjusted_image = self.apply_windowing(image_data, w_width, w_center)

        # Return adjusted image to subject
        subject['image'][tio.DATA] = adjusted_image
        return subject
    
class ContrastStretch(tio.Transform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply_transform(self, subject):
        for image in subject.get_images():
            data = image.data.numpy()

            # 对比度拉伸
            p2, p98 = np.percentile(data, (2, 98))
             # 检查 p2 和 p98 是否相等
            if p98 > p2:
                stretched = (data - p2) / (p98 - p2)  # Normalize to [0, 1]
            else:
                stretched = np.zeros_like(data)  # 如果 p2 和 p98 相等，返回全零数组
            stretched = np.clip(stretched, 0, 1)  # Clip to [0, 1]
            image.data = torch.tensor(stretched).float()

        return subject

class PointPromptGenerator:
    """生成点提示的类"""

    def __init__(self, num_points=6, foreground_ratio=0.7):
        """
        Args:
            num_points: 总点数
            foreground_ratio: 前景点比例
        """
        self.num_points = num_points
        self.foreground_ratio = foreground_ratio
        self.num_foreground = int(num_points * foreground_ratio)
        self.num_background = num_points - self.num_foreground

    def generate_points(self, mask):
        """
        根据ground truth mask生成点提示

        Args:
            mask: 分割mask，形状为 [1, D, H, W] 或 [D, H, W]

        Returns:
            points_coords: 点坐标，形状为 [num_points, 3]，归一化到[0,1]
            points_labels: 点标签，1表示前景，0表示背景，形状为 [num_points]
        """
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask.squeeze()

        # 获取空间维度
        if len(mask_np.shape) == 3:
            D, H, W = mask_np.shape
        else:
            raise ValueError(f"Mask shape should be 3D or 4D, got {mask_np.shape}")

        # 找到前景和背景的坐标
        foreground_coords = np.argwhere(mask_np > 0)  # [N_fg, 3]
        background_coords = np.argwhere(mask_np == 0)  # [N_bg, 3]

        points_coords = []
        points_labels = []

        # 生成前景点
        if len(foreground_coords) > 0:
            # 选择一些随机前景点
            if len(foreground_coords) >= self.num_foreground:
                selected_indices = np.random.choice(
                    len(foreground_coords), self.num_foreground, replace=False
                )
            else:
                selected_indices = np.random.choice(
                    len(foreground_coords), self.num_foreground, replace=True
                )

            for idx in selected_indices:
                coord = foreground_coords[idx]
                # 归一化坐标到 [0, 1]
                normalized_coord = [
                    coord[0] / (D - 1) if D > 1 else 0.5,
                    coord[1] / (H - 1),
                    coord[2] / (W - 1)
                ]
                points_coords.append(normalized_coord)
                points_labels.append(1)  # 前景点标签为1

        # 如果前景点不足，用背景点补足
        while len(points_coords) < self.num_foreground:
            if len(foreground_coords) > 0:
                coord = foreground_coords[np.random.randint(len(foreground_coords))]
            else:
                # 如果没有前景，使用随机坐标
                coord = [np.random.randint(D), np.random.randint(H), np.random.randint(W)]

            normalized_coord = [
                coord[0] / (D - 1) if D > 1 else 0.5,
                coord[1] / (H - 1),
                coord[2] / (W - 1)
            ]
            points_coords.append(normalized_coord)
            points_labels.append(1)

        # 生成背景点
        if len(background_coords) > 0:
            if len(background_coords) >= self.num_background:
                selected_indices = np.random.choice(
                    len(background_coords), self.num_background, replace=False
                )
            else:
                selected_indices = np.random.choice(
                    len(background_coords), self.num_background, replace=True
                )

            for idx in selected_indices:
                coord = background_coords[idx]
                normalized_coord = [
                    coord[0] / (D - 1) if D > 1 else 0.5,
                    coord[1] / (H - 1),
                    coord[2] / (W - 1)
                ]
                points_coords.append(normalized_coord)
                points_labels.append(0)  # 背景点标签为0

        # 如果背景点不足，用随机点补足
        while len(points_coords) < self.num_points:
            coord = [np.random.randint(D), np.random.randint(H), np.random.randint(W)]
            normalized_coord = [
                coord[0] / (D - 1) if D > 1 else 0.5,
                coord[1] / (H - 1),
                coord[2] / (W - 1)
            ]
            points_coords.append(normalized_coord)
            # 如果mask中这个位置是前景，标签为1，否则为0
            label = 1 if mask_np[coord[0], coord[1], coord[2]] > 0 else 0
            points_labels.append(label)

        # 转换为tensor
        points_coords = torch.tensor(points_coords, dtype=torch.float32)
        points_labels = torch.tensor(points_labels, dtype=torch.float32)

        return points_coords, points_labels

class MyDataset_test(Dataset):
    def __init__(self,image_path,label_path, num_points=6, foreground_ratio=0.7):
        self.image_path = image_path
        self.label_path = label_path
        self.name = os.listdir(image_path)
        self.transform = tio.Compose([

        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(128,128,128)), # crop only object region
        # tio.RandomFlip(axes=(0, 1, 2)),
        ########################################## data processing
        CustomTransform(),
        tio.ZNormalization(),
        tio.Resample(),
        # ContrastStretch(),
        ##########################################
    ])
        # self.prompt_generator = PointPromptGenerator(num_points, foreground_ratio)

    def __getitem__(self, index):

        im_path = os.path.join(self.image_path,self.name[index])
        # la_path = os.path.join(self.label_path,self.name[index])
        la_path = im_path.replace("image", "label")
        # print(im_path)

        ct = sitk.ReadImage(im_path)
        # print(ct.GetSize())
        seg = sitk.ReadImage(la_path)
        # print(seg.GetSize())
        if ct.GetOrigin() != seg.GetOrigin():
            ct.SetOrigin(seg.GetOrigin())
        if ct.GetDirection() != seg.GetDirection():
            ct.SetDirection(seg.GetDirection())

        ct_array, _ = sitk_to_nib(ct)
        seg_array, _ = sitk_to_nib(seg)
        seg_array[seg_array > 0] = 1
        # print(type(seg_array))
        # print(seg_array.shape)
        H,W = seg_array.shape[2],seg_array.shape[3]
        # se_array = seg_array.squeeze(0)
        # se_array = se_array[1,:,:]
        # cv2.imwrite(r"D:\fast\hzh-SAM-Med3D\hzh.png",se_array)

        # seg_array[seg_array==255] = 1
        # print(np.unique(seg_array))
        subject = tio.Subject(
            image = tio.ScalarImage(tensor=ct_array),
            label = tio.LabelMap(tensor=seg_array)
        )
        subject = self.transform(subject)
        # 生成点提示
        # points_coords, points_labels = self.prompt_generator.generate_points(
        #     subject.label.data
        # )
        # seg_array = np.transpose(seg_array,(2,0,1))
        # seg_array = np.expand_dims(seg_array, axis=0)
        # seg_array = np.tile(seg_array, (3,1,1,1))
        # print(seg_array.shape)

        # print(np.unique(seg_array))
        # print(subject.label.data.shape)
        meta_info = {
            "size": (H,W),
            "path": im_path,
            "direction": ct.GetDirection(),
            "origin": ct.GetOrigin(),
            "spacing": ct.GetSpacing(),
        }
        return (subject.image.data.clone().detach().float(),
                subject.label.data.clone().detach().float(), meta_info)

    def __len__(self):
        return len(self.name)

if __name__ == '__main__':
    im_path = r"D:\all_data\nii\image"
    la_path = r"D:\all_data\nii\label"

    MyDataset = MyDataset(image_path=im_path, label_path=la_path)
    # MyDataset[0]

    loader = DataLoader(MyDataset, batch_size=1, shuffle=True)

    for index, (img, mask) in enumerate(tqdm(loader,total=len(loader))):
        break