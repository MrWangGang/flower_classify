import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import io
import os

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
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# --- 结合 CBAM 的 ResNet 模型 ---
class ResNetWithCBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNetWithCBAM, self).__init__()
        resnet = models.resnet152(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.cbam = CBAM(in_planes=2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- 普通 ResNet 模型 ---
def get_resnet_model(num_classes):
    model = models.resnet152(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# 全局变量和路径
NUM_CLASSES = 102
LABEL_PATH = './datasets/label.json'
RESNET_WEIGHTS_PATH = './report/resnet/best_model.pth'
CBAM_WEIGHTS_PATH = './report/attention/best_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR_FOR_MAPPING = './datasets/train'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_models():
    resnet_model = get_resnet_model(NUM_CLASSES)
    resnet_model.load_state_dict(torch.load(RESNET_WEIGHTS_PATH, map_location=DEVICE))
    resnet_model.to(DEVICE)
    resnet_model.eval()

    cbam_model = ResNetWithCBAM(NUM_CLASSES)
    cbam_model.load_state_dict(torch.load(CBAM_WEIGHTS_PATH, map_location=DEVICE))
    cbam_model.to(DEVICE)
    cbam_model.eval()

    for param in resnet_model.parameters():
        param.requires_grad = False
    for param in cbam_model.parameters():
        param.requires_grad = False

    return resnet_model, cbam_model

@st.cache_resource
def load_labels():
    try:
        with open(LABEL_PATH, 'r', encoding='utf-8') as f:
            class_names_map = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error(f"错误: 无法加载标签文件 {LABEL_PATH}，或文件格式错误。")
        st.stop()

    if not os.path.isdir(DATA_DIR_FOR_MAPPING):
        st.error(f"错误: 文件夹 {DATA_DIR_FOR_MAPPING} 不存在，无法创建类别映射。")
        st.stop()

    folder_names = sorted(os.listdir(DATA_DIR_FOR_MAPPING))

    idx_to_label = {}
    for idx, folder_name in enumerate(folder_names):
        if folder_name in class_names_map:
            idx_to_label[idx] = class_names_map[folder_name]
        else:
            idx_to_label[idx] = folder_name

    return idx_to_label

def predict(model, image, idx_to_label):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class_idx = predicted.item()

    if predicted_class_idx in idx_to_label:
        predicted_class_label = idx_to_label[predicted_class_idx]
    else:
        predicted_class_label = f"未知类别 (索引: {predicted_class_idx})"

    confidence_score = confidence.item()

    return predicted_class_label, confidence_score

# --- Streamlit 界面 ---
st.set_page_config(page_title="模型图片分类演示", layout="wide")

st.title("图像分类模型演示")
st.markdown("请选择您想要测试的图片，并上传给其中一个模型进行分类。")
st.markdown("---")

# 加载模型和标签
resnet_model, cbam_model = load_models()
idx_to_label = load_labels()

col1, col2 = st.columns(2)

with col1:
    # 在第一列内部再创建两个子列，用于标题和结果
    title_col, result_col = st.columns([0.6, 0.4])
    with title_col:
        st.subheader("普通 ResNet-152 模型")

    uploaded_file_resnet = st.file_uploader("上传图片给 ResNet", type=["jpg", "jpeg", "png"], key="resnet")
    if uploaded_file_resnet:
        image = Image.open(io.BytesIO(uploaded_file_resnet.getvalue())).convert("RGB")
        st.image(image, caption="ResNet 上传的图片", width=300)
        with st.spinner("正在使用 ResNet-152 模型进行预测..."):
            label, confidence = predict(resnet_model, image, idx_to_label)

        # 将结果输出到右侧的列中
        with result_col:
            st.success(f"分类结果：**{label}**")
            st.info(f"置信度: **{confidence:.2%}**")

with col2:
    # 在第二列内部再创建两个子列，用于标题和结果
    title_col, result_col = st.columns([0.6, 0.4])
    with title_col:
        st.subheader("CBAM ResNet-152 模型")

    uploaded_file_cbam = st.file_uploader("上传图片给 CBAM ResNet", type=["jpg", "jpeg", "png"], key="cbam")
    if uploaded_file_cbam:
        image = Image.open(io.BytesIO(uploaded_file_cbam.getvalue())).convert("RGB")
        st.image(image, caption="CBAM ResNet 上传的图片", width=300)
        with st.spinner("正在使用 CBAM ResNet-152 模型进行预测..."):
            label, confidence = predict(cbam_model, image, idx_to_label)

        # 将结果输出到右侧的列中
        with result_col:
            st.success(f"分类结果：**{label}**")
            st.info(f"置信度: **{confidence:.2%}**")