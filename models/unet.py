import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(1, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.b = DoubleConv(256, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)

        b = self.b(p3)

        u3 = self.u3(b)
        c3 = self.c3(torch.cat([u3, d3], dim=1))

        u2 = self.u2(c3)
        c2 = self.c2(torch.cat([u2, d2], dim=1))

        u1 = self.u1(c2)
        c1 = self.c1(torch.cat([u1, d1], dim=1))

        return torch.sigmoid(self.out(c1))
