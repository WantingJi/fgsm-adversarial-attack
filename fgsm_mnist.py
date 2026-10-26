import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 一个简单的 MNIST 分类器
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 加载 MNIST 数据集
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./datasets', train = False, download=True, 
                                                         transform=transforms.ToTensor()),
                                            batch_size = 1, shuffle=True)
device = torch.device("cpu")
model = SimpleNet().to(device)

# 快速简单训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./datasets', train=True, download=True, 
                                                         transform=transforms.ToTensor()),
                                            batch_size=64, shuffle=True)

model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()


# FGSM 攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

# 测试干净样本和对抗样本的准确率
model.eval()
epsilon = 0.15
clean_correct, adv_correct, total = 0, 0, 0

for image, label in test_loader:
    image, label = image.to(device), label.to(device)
    image.requires_grad = True

    output = model(image)
    clean_correct += (output.argmax(1) == label).item()

    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    adv_image = fgsm_attack(image, epsilon, image.grad)
    adv_output = model(adv_image)
    adv_correct += (adv_output.argmax(1) == label).item()
    total += 1

print(f"Clean Accuracy: {clean_correct/total*100:.1f}%")
print(f"FGSM Accuracy (epsilon={epsilon}): {adv_correct/total*100:.1f}%")