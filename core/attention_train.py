import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve


# --- 配置参数 ---
DATA_DIR = './datasets'
MODEL_WEIGHTS_PATH = './model/resnet/resnet152-b121ed2d.pth'
LABEL_JSON_PATH = './datasets/label.json'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = './report/attention'

class AddGaussianNoise(object):
    """
    向图像添加高斯噪声，并可选择以一定的概率应用。
    """
    def __init__(self, mean=0., std=1., p=1.0):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, p={self.p})'

class AddSaltAndPepperNoise(object):
    """
    向图像添加椒盐噪声，并可选择以一定的概率应用。
    """
    def __init__(self, probability=0.01, p=1.0):
        self.probability = probability
        self.p = p

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError('This transform only works on torch.Tensor images.')

        if random.random() < self.p:
            c, h, w = img.shape
            salt_pixels = torch.rand(h, w) < self.probability / 2
            pepper_pixels = torch.rand(h, w) < self.probability / 2

            img[:, salt_pixels] = 1.0  # Salt noise (white)
            img[:, pepper_pixels] = 0.0  # Pepper noise (black)

        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(probability={self.probability}, p={self.p})'

class AddMedianBlur(object):
    """
    使用中值滤波去除椒盐噪声。
    这个自定义类使用PIL的MedianFilter。
    """
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError('This transform only works on torch.Tensor images.')

        img_pil = transforms.ToPILImage()(img)
        img_pil = img_pil.filter(ImageFilter.MedianFilter(self.kernel_size))
        return transforms.ToTensor()(img_pil)

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'


# --- 数据增强和加载器 (保持不变) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
        transforms.ToTensor(),
        AddSaltAndPepperNoise(probability=0.01, p=0.5),
        AddGaussianNoise(mean=0., std=0.03, p=0.5),
        AddMedianBlur(kernel_size=3),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

visual_augment_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    AddSaltAndPepperNoise(probability=0.01, p=1.0),
    AddGaussianNoise(mean=0., std=0.03, p=1.0),
])

raw_image_transforms = transforms.Compose([
    transforms.ToTensor()
])
raw_image_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), raw_image_transforms)
image_datasets = {
    x: ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'valid']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    for x in ['train', 'valid']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
noisy_image_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), visual_augment_transforms)

with open(LABEL_JSON_PATH, 'r', encoding='utf-8') as f:
    class_names_map = json.load(f)

class_to_idx = image_datasets['train'].class_to_idx
num_classes = len(class_to_idx)
class_names = [None] * num_classes
for class_folder_name, idx in class_to_idx.items():
    class_names[idx] = class_names_map[class_folder_name]

print(f"Detected {num_classes} classes: {class_names}")

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

augment_dir = os.path.join(REPORT_DIR, 'augment')
if not os.path.exists(augment_dir):
    os.makedirs(augment_dir)

unshuffled_train_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'),
                                       transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
unshuffled_dataloader = DataLoader(unshuffled_train_dataset, batch_size=1)

num_examples = 20
sample_indices = random.sample(range(len(unshuffled_train_dataset)), num_examples)

print(f"Generating {num_examples} augmented image examples...")
for i, idx in enumerate(sample_indices):
    original_raw_image, _ = raw_image_dataset[idx]
    noisy_image, _ = noisy_image_dataset[idx]
    original_image_for_display = transforms.Resize((224, 224))(original_raw_image)

    original_image_for_display = torch.clamp(original_image_for_display, 0, 1)
    noisy_image = torch.clamp(noisy_image, 0, 1)

    original_pil = transforms.ToPILImage()(original_image_for_display)
    noisy_pil = transforms.ToPILImage()(noisy_image)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(original_pil)
    axes[0].set_title('Original Image (Resized)')
    axes[0].axis('off')

    axes[1].imshow(noisy_pil)
    axes[1].set_title('Augmented & Noisy Image')
    axes[1].axis('off')

    plt.suptitle(f"Image Augmentation Example {i+1}")
    plt.tight_layout()

    image_path, label_idx = unshuffled_train_dataset.imgs[idx]
    original_filename = os.path.basename(image_path)
    label_name = class_names[label_idx]

    base_filename = os.path.splitext(original_filename)[0]
    new_filename = f"{base_filename}_{label_name}.png"

    plt.savefig(os.path.join(augment_dir, new_filename))
    plt.close()

print(f"Augmented image examples saved in {augment_dir}")


# --- CBAM 注意力机制模块 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = self.ca(x) * x
        # 空间注意力
        x = self.sa(x) * x
        return x


# --- 结合 CBAM 的 ResNet 模型 ---
class ResNetWithCBAM(nn.Module):
    def __init__(self, model_weights_path, num_classes):
        super(ResNetWithCBAM, self).__init__()

        # 加载预训练的 ResNet-152 模型
        resnet = models.resnet152(weights=None)
        state_dict = torch.load(model_weights_path, map_location='cpu', weights_only=False)
        resnet.load_state_dict(state_dict)

        # 移除 ResNet 的最后两层，只保留特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 冻结特征提取部分的参数
        for param in self.features.parameters():
            param.requires_grad = False

        # 添加 CBAM 注意力模块
        self.cbam = CBAM(in_planes=2048) # ResNet-152 最后一层的通道数是 2048

        # 添加全局平均池化层和新的全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # 通过 ResNet 特征提取部分
        x = self.features(x)

        # 在全局池化前应用 CBAM
        x = self.cbam(x)

        # 进行全局平均池化和分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNetWithCBAM(MODEL_WEIGHTS_PATH, num_classes).to(DEVICE)
print("ResNet-152 model with CBAM loaded and modified.")

# 确保优化器只更新需要训练的参数 (fc 和 attention 层)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)


# --- 训练和评估函数 (保持不变) ---
criterion = nn.CrossEntropyLoss()

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'accuracy_curve.png'))
    plt.close()

    metrics = ['f1', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [h[f'macro_{metric}'] if f'macro_{metric}' in h else np.nan for h in history['per_class_metrics']], label=f'Macro-average {metric.capitalize()}')
        plt.title(f'Overall Macro-average {metric.capitalize()} Scores')
        plt.xlabel('Epochs')
        plt.ylabel(f'Macro-average {metric.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(REPORT_DIR, f'macro_{metric}_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    if not y_true:
        print("Skipping Confusion Matrix plot: No validation data available.")
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(100, 100))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

def plot_roc_pr_curves(y_true, y_scores, num_classes, class_names):
    if len(y_true) == 0 or len(y_scores) == 0:
        print("Skipping ROC/PR Curves plot: No validation data available.")
        return
    y_true_one_hot = np.zeros((len(y_true), num_classes))
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_roc_auc = auc(macro_fpr, macro_tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(macro_fpr, macro_tpr, label=f'Macro-average ROC curve (AUC = {macro_roc_auc:.2f})', color='deeppink', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'roc_curve.png'))
    plt.close()

    precision = dict()
    recall = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_one_hot[:, i], y_scores[:, i])

    plt.figure(figsize=(10, 6))
    all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(num_classes):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= num_classes

    plt.plot(all_recall, mean_precision, color='deeppink', linestyle=':', linewidth=4, label='Macro-average PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'pr_curve.png'))
    plt.close()

def train_model(model, criterion, optimizer, num_epochs=EPOCHS):
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'per_class_metrics': []}
    best_accuracy = 0.0

    all_labels_final = []
    all_preds_final = []
    all_outputs_final = []

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")

        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1} [Train]"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / dataset_sizes['train'] if dataset_sizes['train'] > 0 else np.nan
        history['train_loss'].append(epoch_train_loss)
        print(f"Training Loss: {epoch_train_loss:.4f}")

        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_outputs = []

        if dataset_sizes['valid'] > 0:
            with torch.no_grad():
                for inputs, labels in tqdm(dataloaders['valid'], desc=f"Epoch {epoch+1} [Valid]"):
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    running_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())

            epoch_val_loss = running_loss / dataset_sizes['valid']
            history['val_loss'].append(epoch_val_loss)
            print(f"Validation Loss: {epoch_val_loss:.4f}")

            report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
            overall_metrics = {
                'macro_f1': report['macro avg']['f1-score'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
            }
            history['per_class_metrics'].append(overall_metrics)

            overall_accuracy = report['accuracy']
            history['val_accuracy'].append(overall_accuracy)
            print(f"Validation Accuracy: {overall_accuracy:.4f}")
            print("\nValidation Class Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

            if overall_accuracy > best_accuracy:
                best_accuracy = overall_accuracy
                best_model_path = os.path.join(REPORT_DIR, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")

            all_labels_final = all_labels
            all_preds_final = all_preds
            all_outputs_final = all_outputs
        else:
            print("Validation dataset is empty, skipping validation for this epoch.")
            history['val_loss'].append(np.nan)
            history['per_class_metrics'].append({'macro_f1': np.nan, 'macro_precision': np.nan, 'macro_recall': np.nan})
            history['val_accuracy'].append(np.nan)

    print("Training complete!")

    print("Generating plots...")
    plot_metrics(history)
    plot_confusion_matrix(all_labels_final, all_preds_final, class_names)
    plot_roc_pr_curves(np.array(all_labels_final), np.array(all_outputs_final), num_classes, class_names)
    print(f"Plots saved in {REPORT_DIR}")
    print(f"The best model weights are saved at: {os.path.join(REPORT_DIR, 'best_model.pth')}")

train_model(model, criterion, optimizer)