import torch
from torch import nn


class _AlexNet(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 128, 3, 1, groups=2))



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.f = AlexNetV1()
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


        self.fc1 = nn.Linear(128 * 2, 512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 2)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.l = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

    def forward(self, x, y):
        f1 = self.flatten(self.f(x))
        f2 = self.flatten(self.f(y))

        z = torch.concat([f1, f2], 1)
        out = self.l(z)
        # out = self.cos(f1, f2)


        return out

    def feature(self, x):
        return self.flatten(self.f(x))

    def ev(self, f1, f2):
        # out = self.cos(f1, f2)
        z1 = torch.cat([f1, f2], 1)
        out1 = self.l(z1)
        out1 = torch.softmax(out1, 1)[:, 1]

        z2 = torch.cat([f2, f1], 1)
        out2 = self.l(z2)
        out2 = torch.softmax(out2, 1)[:, 1]
        # out = torch.argmax(out, 1)
        # m = out.item()
        return torch.max(out1, out2)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeuralNetwork().to(device)
    print(model)