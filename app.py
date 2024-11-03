from flask import Flask, render_template, jsonify
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch.nn as nn

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình và chuẩn bị dữ liệu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Định nghĩa các hàm và tải dữ liệu
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

# Dữ liệu kiểm thử
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Hàm chuyển đổi ảnh sang base64
def tensor_to_base64(img_tensor):
    img_tensor = img_tensor / 2 + 0.5  # Đưa ảnh về khoảng giá trị [0, 1]
    np_img = img_tensor.squeeze().permute(1, 2, 0).numpy()  # Đảm bảo ảnh ở định dạng HxWxC
    pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    return model

# Khởi tạo mô hình và nạp trọng số đã lưu
n_features = 32 * 32 * 3  # Kích thước của ảnh CIFAR-10
model = getModel(n_features)
model = torch.load('MLP_dress.pth', map_location=device)
model.eval()  # Chuyển mô hình sang chế độ đánh giá để dự đoán


# Tạo route cho trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# API để lấy ảnh mẫu và dự đoán
@app.route('/predict', methods=['GET'])
def predict():
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    img_base64 = tensor_to_base64(images[0].cpu())  # Chuyển đổi ảnh sang base64
    prediction = {
        'image': img_base64,  # Trả về chuỗi base64 của ảnh
        'predicted_class': predicted.item()
    }
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
