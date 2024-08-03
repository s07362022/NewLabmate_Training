import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Set a specific seed value, you can change this to any integer
# set_seed(69)

# 定義U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block

        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.final_conv(dec1)
        # return F.softmax(self.final_conv(dec1), dim=1)

# 創建一個簡單的自定義數據集類（此處假設你會使用自己的數據集）
class SimpleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 掩码通常是灰度图
        

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = mask.squeeze(0)
        mask = torch.round(mask * 255).long()
        # print(np.unique(mask))
        return image, mask

# 定義 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)  # 使用softmax將輸出轉換為概率
        num_classes = preds.size(1)

        # 将 targets 转换为 long 类型，然后进行 one-hot 编码
        targets_one_hot = torch.eye(num_classes)[targets.squeeze(1).long().cpu()].to(preds.device)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 计算每个类的 Dice Loss
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice_score

# 計算 F1-score
def calculate_f1(pred, target, num_classes):

    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()


    # 计算 F1-score
    # print('t',np.unique(target_flat))
    # print('p',np.unique(pred_flat))
    return f1_score(target_flat, pred_flat, average='macro')


# 繪製結果圖像
def plot_results(images, masks, predictions, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(len(images), 5)):  # 只顯示前5張
        image = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i].cpu().numpy()
        pred = torch.argmax(predictions[i], dim=0).cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title('Input Image')
        ax[1].imshow(mask)
        ax[1].set_title('True Mask')
        ax[2].imshow(pred)
        ax[2].set_title('Predicted Mask')

        plt.savefig(os.path.join(output_dir, f'result_epoch_{epoch}_image_{i}.png'))
        plt.close()

# 計算 MIOU 的函數
def calculate_miou(pred, target, num_classes):
    ious = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = torch.sum(pred_inds & target_inds).item()
        union = torch.sum(pred_inds | target_inds).item()
        if union == 0:
            ious.append(float('nan'))  # 如果類別未出現，則忽略
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # 返回 MIOU 的平均值

# 定義訓練的參數
in_channels = 3       # 輸入圖像的通道數（RGB圖像為3）
out_channels = 3      # 輸出的類別數（根據你的數據集進行調整）
batch_size = 2
learning_rate = 1e-5
num_epochs = 10

# 創建數據增強和數據加載器
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(0.5),               # 隨機水平翻轉
    transforms.RandomVerticalFlip(0.5),                 # 隨機垂直翻轉
    transforms.RandomRotation(30),                   # 隨機旋轉
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 顏色抖動
    # transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),  # 隨機裁剪和調整
    # transforms.RandomAffine(degrees=30, shear=0.2),  # 隨機仿射變換
    transforms.ToTensor(), 
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 資料夾路徑
train_image_dir = r'0615_patch'  # 替換為你的訓練圖像資料夾路徑
train_mask_dir = r'0615_mix'     # 替換為你的訓練掩碼資料夾路徑

val_image_dir = r'0614_patch'    # 替換為你的驗證圖像資料夾路徑
val_mask_dir = r'0614_mix'         # 替換為你的驗證掩碼資料夾路徑
test_image_dir = r'patch_20'
test_mask_dir = r'p_20_mix'


# 創建訓練和驗證數據集和加載器
train_dataset = SimpleDataset(train_image_dir, train_mask_dir, transform=train_transform)
val_dataset = SimpleDataset(val_image_dir, val_mask_dir, transform=val_transform)
test_dataset = SimpleDataset(test_image_dir, test_mask_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、損失函數和優化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels, out_channels).to(device)
# criterion = nn.CrossEntropyLoss()
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

# 訓練和驗證模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_miou = 0.0

    # 使用tqdm顯示訓練進度
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for images, targets in train_loader_tqdm:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        soft_pro = F.softmax(outputs, dim=1)
        # loss = criterion(outputs, targets.long())
        loss = criterion(soft_pro, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_miou += calculate_miou(outputs, targets, out_channels)

        # 更新tqdm進度條信息
        train_loader_tqdm.set_postfix({"Loss": f"{loss:.4f}", "MIOU": f"{(running_miou / (train_loader_tqdm.n + 1)):.4f}"})

    epoch_loss = running_loss / len(train_loader)
    epoch_miou = running_miou / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training MIOU: {epoch_miou:.4f}')

    # 驗證模型
    model.eval()
    val_running_miou = 0.0
    val_running_f1 = 0.0
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = val_images.to(device)
            val_targets = val_targets.to(device)

            val_outputs = model(val_images)
            val_preds = torch.argmax(val_outputs, dim=1)  # 这里使用argmax获取预测的类别标签
            val_running_miou += calculate_miou(val_outputs, val_targets, out_channels)
            val_running_f1 += calculate_f1(val_preds, val_targets, out_channels)
            # print(val_preds.shape, val_targets.shape)

    val_epoch_miou = val_running_miou / len(val_loader)
    val_epoch_f1 = val_running_f1 / len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation MIOU: {val_epoch_miou:.4f}, Validation F1-score: {val_epoch_f1:.4f}')

    

# 保存模型
torch.save(model.state_dict(), 'unet_custom.pth')


# 測試模型
test_running_miou = 0.0
test_running_f1 = 0.0
with torch.no_grad():
    for test_images, test_targets in test_loader:
        test_images = test_images.to(device)
        test_targets = test_targets.to(device)

        test_outputs = model(test_images)
        test_preds = torch.argmax(test_outputs, dim=1)  # 这里使用argmax获取预测的类别标签
        test_running_miou += calculate_miou(test_outputs, test_targets, out_channels)
        test_running_f1 += calculate_f1(test_preds, test_targets, out_channels)

# 計算最終的MIOU和F1-score
test_miou = test_running_miou / len(test_loader)
test_f1 = test_running_f1 / len(test_loader)
print(f'Test MIOU: {test_miou:.4f}, Test F1-score: {test_f1:.4f}')


plot_results(test_images, test_targets, test_outputs, output_dir='test_results', epoch='final', show=True)






