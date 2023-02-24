import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        num_heads = 1
        dim = 512
        qk_scale = None
        qkv_bias = True
        attn_drop_ratio = 0.
        proj_drop_ratio = 0.
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.block3 = GAM_Attention(512, 512)
        self.block4 = simam_module()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x  # torch.Size([64, 512, 6, 6])

        # GAM
        out = self.block3(out)
        # simam_module
        out = self.block4(out)

        B, C, W, H = out.shape
        x = out.permute(0, 2, 3, 1)
        x = x.view(B, W * H, C)

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        x = x.view(B, W, H, C)
        x = x.permute(0, 3, 2, 1)

        out = self.proj_drop(x)

        return out


# Loading the vgg16 pre-trained model
vgg16 = torchvision.models.vgg16(pretrained=True)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = []

        # The first convolution part loads the first block in the vgg16 model
        # 112, 112, 64
        vgg.append(vgg16.features[0])
        vgg.append(vgg16.features[1])
        vgg.append(vgg16.features[2])
        vgg.append(vgg16.features[3])
        vgg.append(vgg16.features[4])

        # The second convolution part loads the second block in the vgg16 model
        # 56, 56, 128
        vgg.append(vgg16.features[5])
        vgg.append(vgg16.features[6])
        vgg.append(vgg16.features[7])
        vgg.append(vgg16.features[8])
        vgg.append(vgg16.features[9])

        # The third convolution part loads the third block in the vgg16 model
        # 28, 28, 256
        vgg.append(vgg16.features[10])
        vgg.append(vgg16.features[11])
        vgg.append(vgg16.features[12])
        vgg.append(vgg16.features[13])
        vgg.append(vgg16.features[14])
        vgg.append(vgg16.features[15])
        vgg.append(vgg16.features[16])

        # The fourth convolution part loads the fourth block in the vgg16 model
        # 14, 14, 512
        vgg.append(vgg16.features[17])
        vgg.append(vgg16.features[18])
        vgg.append(vgg16.features[19])
        vgg.append(vgg16.features[20])
        vgg.append(vgg16.features[21])
        vgg.append(vgg16.features[22])
        vgg.append(vgg16.features[23])

        # The fifth convolution part loads the fifth block in the vgg16 model
        # 7, 7, 512
        vgg.append(vgg16.features[24])
        vgg.append(vgg16.features[25])
        vgg.append(SelfAttention(512))  # selfattention
        vgg.append(vgg16.features[26])
        vgg.append(vgg16.features[27])
        vgg.append(SelfAttention(512))  # selfattention
        vgg.append(vgg16.features[28])
        vgg.append(vgg16.features[29])
        vgg.append(vgg16.features[30])
        vgg.append(SelfAttention(512))  # selfattention

        # Send each module into nn.Sequential according to their order, input either orderdict or a series of models, when encountering the above list, you must use * to convert
        self.main = nn.Sequential(*vgg)

        self.avgpool = vgg16.avgpool

        # Fully connected layer Add the fully connected layer in the vgg16 model
        classfication = []
        # in_features four-dimensional tensor becomes two-dimensional [batch_size, channels, width, height] becomes [batch_size, channels*width*height]
        classfication.append(vgg16.classifier[0])
        classfication.append(vgg16.classifier[1])
        classfication.append(vgg16.classifier[2])
        classfication.append(vgg16.classifier[3])
        classfication.append(vgg16.classifier[4])
        classfication.append(vgg16.classifier[5])
        # classfication.append(vgg16.classifier[6])
        classfication.append(nn.Linear(in_features=4096, out_features=6, bias=True))

        self.classfication = nn.Sequential(*classfication)

    def forward(self, x):
        feature = self.main(x)  # input tensor x
        # print(feature.shape)
        feature = self.avgpool(feature)
        feature = feature.view(x.size(0), -1)  # reshape x becomes [batch_size, channels*width*height]
        result = self.classfication(feature)
        return result

def VGG16(pretrained, in_channels, **kwargs):
    model = VGG()
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model

# GAM
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

# simAM
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)