from torch import nn
import torch.nn.functional as F


class AECLS(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.enc_bn_1 = nn.BatchNorm2d(32)
        self.enc_cnn_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.enc_bn_2 = nn.BatchNorm2d(64)
        self.enc_cnn_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.enc_bn_3 = nn.BatchNorm2d(128)
        self.enc_cnn_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.enc_bn_4 = nn.BatchNorm2d(256)
        self.enc_linear_1 = nn.Linear(30976, 2048)
        self.dropout_0 = nn.Dropout(0.2)
        self.enc_bn_5 = nn.BatchNorm1d(2048)
        # Decoder specification
        self.dec_decnn_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.dec_bn_1 = nn.BatchNorm2d(128)
        self.dec_decnn_2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.dec_bn_2 = nn.BatchNorm2d(64)
        self.dec_decnn_3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.dec_bn_3 = nn.BatchNorm2d(32)
        self.dec_decnn_4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2)

        self.dec_linear_1 = nn.Linear(2048, 30976)
        self.dec_bn_0 = nn.BatchNorm1d(30976)

        self.cls_linear_1 = nn.Linear(2048, 512)
        self.dropout_1 = nn.Dropout(0.2)
        self.cls_bn_0 = nn.BatchNorm1d(512)
        self.cls_linear_2 = nn.Linear(512, 27)

    def forward(self, images):
        code = self.encode(images)
        out_rec = self.decode(code)
        out_cls = self.cls(code)

        return [out_rec, out_cls]

    def encode(self, images):
        code = F.relu(self.enc_cnn_1(images))
        code = self.enc_bn_1(code)

        code = F.relu(self.enc_cnn_2(code))
        code = self.enc_bn_2(code)

        code = F.relu(self.enc_cnn_3(code))
        code = self.enc_bn_3(code)

        code = F.relu(self.enc_cnn_4(code))
        code = self.enc_bn_4(code)

        code = code.view([code.size(0), code.size(1) * code.size(2) * code.size(3)])
        code = self.enc_linear_1(code)
        #code = self.dropout_0(code)
        code = self.enc_bn_5(code)
        return code

    def decode(self, code):
        out = self.dec_linear_1(code)
        out = self.dec_bn_0(out)
        out = out.view([out.size(0), -1, 11, 11])
        out = F.relu(self.dec_decnn_1(out))
        out = self.dec_bn_1(out)
        out = F.relu(self.dec_decnn_2(out))
        out = self.dec_bn_2(out)
        out = F.relu(self.dec_decnn_3(out))
        out = self.dec_bn_3(out)
        out = self.dec_decnn_4(out)
        return out

    def cls(self, code):
        out = F.relu(self.cls_linear_1(code))
        out = self.cls_bn_0(out)
        out = self.cls_linear_2(out)
        return out