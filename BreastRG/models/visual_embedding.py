import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class up_conv_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class resconv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.GroupNorm(8, ch_out),
            #nn.InstanceNorm3d(ch_out),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True),
            #nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.GroupNorm(8, ch_out),
            #nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)

        return  x + residual


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)

        return x

class resconv_block_3D_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.BatchNorm3d(ch_out)
            nn.BatchNorm3d(8, ch_out),
            nn.ReLU(inplace = True),
            #nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            #nn.GroupNorm(8, ch_out),
            #nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)
        return residual + x

class TABS(nn.Module):
    def __init__(
        self,
        img_dim = 192,
        patch_dim = 8,
        img_ch = 6,
        output_ch = 3,
        embedding_dim = 512,
        num_heads = 8,
        num_layers = 4,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        ):
        super(TABS,self).__init__()

        self.hidden_dim = int((img_dim/16)**3)
        self.hidden_dim = 2048
        self.Maxpool = nn.MaxPool3d(kernel_size=(3,3,2),stride=2)

        self.Conv1 = resconv_block_3D(ch_in=img_ch,ch_out=8)

        self.Conv2 = resconv_block_3D(ch_in=8,ch_out=16)
        self.Conv3 = resconv_block_3D(ch_in=16,ch_out=32)
        self.Conv4 = resconv_block_3D(ch_in=32,ch_out=64)
        self.Conv5 = resconv_block_3D(ch_in=64,ch_out=128)
         
        self.Conv1_1 = resconv_block_3D(ch_in=1,ch_out=8)
        self.Conv2_1 = resconv_block_3D(ch_in=8,ch_out=16)
        self.Conv3_1 = resconv_block_3D(ch_in=16,ch_out=32)
        self.Conv4_1 = resconv_block_3D(ch_in=32,ch_out=64)
        self.Conv5_1 = resconv_block_3D(ch_in=64,ch_out=128)
        
        self.Conv1_2 = resconv_block_3D(ch_in=1,ch_out=8)
        self.Conv2_2 = resconv_block_3D(ch_in=8,ch_out=16)
        self.Conv3_2 = resconv_block_3D(ch_in=16,ch_out=32)
        self.Conv4_2 = resconv_block_3D(ch_in=32,ch_out=64)
        self.Conv5_2 = resconv_block_3D(ch_in=64,ch_out=128)
        '''
        self.Up5 = up_conv_3D(ch_in=128,ch_out=64)
        self.Up_conv5 = resconv_block_3D(ch_in=128, ch_out=64)
        self.Up4 = up_conv_3D(ch_in=64,ch_out=32)
        self.Up_conv4 = resconv_block_3D(ch_in=64, ch_out=32)
        self.Up3 = up_conv_3D(ch_in=32,ch_out=16)
        self.Up_conv3 = resconv_block_3D(ch_in=32, ch_out=16)
        self.Up2 = up_conv_3D(ch_in=16,ch_out=8)
        self.Up_conv2 = resconv_block_3D(ch_in=16, ch_out=8)
        '''
        self.bn = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(128)
        #self.gn = nn.GroupNorm(8, 128)
        #nn.GroupNorm(8, ch_out),
            #nn.ReLU(inplace = True),
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True) 
        self.relu3 = nn.ReLU(inplace=True)
        #self.position_encoding = LearnedPositionalEncoding(
        #    embedding_dim, self.hidden_dim
        #)
        #self.reshaped_conv = conv_block_3D(512, 128)

        self.conv_x = nn.Conv3d(
            128,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.conv_x1 = nn.Conv3d(
            128,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.conv_x2 = nn.Conv3d(
            128,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        #self.type_embedding = nn.Embedding(3, 512)
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.embedding_dim = embedding_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x, x1, x2, mask=None):
        #x: DCE , x1: DWI, x3: T2
        # encoding path
        #b,c,w,h,d = x.shape
        b = x.size(0)
        ids_keep = None
        if mask is None:
            x = x.reshape(shape=(b, 6, 2, 168, 56*2*2, 8, 16))
            x = torch.einsum('bcnwhde->bndcwhe',x)

            
            x = x.reshape(shape=(b, 16, 6, 168, 56*2*2, 16))
            x = x.reshape(shape=(b*16, 6, 168, 56*2*2, 16))
        else:
            ids = torch.argsort(mask.long(), dim=1)  # ascend
            mask_len = mask[0].sum()
            ids_keep = ids[:, : ids.shape[1] - mask_len]

            x = x.reshape(shape=(b, 6, 11, 32, 7, 32, 4, 32))
            x = torch.einsum('bcwmhnde->bwhdcmne',x)
            x = x.reshape(shape=(b,11*7*4, 6, 32, 32, 32))
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 6, 32, 32, 32))
            x = x.reshape(shape=(-1, 6, 32, 32, 32)) 
            #x = rearrange(x, 'b l c w h d -> (b l) c w h d').contiguous()
        
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = self.Conv4(x)
        x = self.Maxpool(x)
        
        x = self.Conv5(x)
        
        x1 = self.Conv1_1(x1)
        x1 = self.Maxpool(x1)
        x1 = self.Conv2_1(x1)
        x1 = self.Maxpool(x1)
        x1 = self.Conv3_1(x1)
        x1 = self.Maxpool(x1)
        x1 = self.Conv4_1(x1)
        x1 = self.Maxpool(x1)
        x1 = self.Conv5_1(x1)
        
        x2 = self.Conv1_2(x2)
        x2 = self.Maxpool(x2)
        x2 = self.Conv2_2(x2)
        x2 = self.Maxpool(x2)
        x2 = self.Conv3_2(x2)
        x2 = self.Maxpool(x2)
        x2 = self.Conv4_2(x2)
        x2 = self.Maxpool(x2)
        x2 = self.Conv5_2(x2)

        x = self.bn(x)
        x1 = self.bn2(x1)
        x2 = self.bn3(x2)

        x = self.relu(x)
        x1 = self.relu2(x1)
        x2 = self.relu3(x2)
        x = self.conv_x(x)
        x1 = self.conv_x1(x1)
        x2 = self.conv_x2(x2)
        x1 = rearrange(x1, 'b c w h d  -> b (w h d) c', b=b).contiguous()
        x = rearrange(x, '(b l) c w h d  -> b (l w h d) c', b=b).contiguous()
        x2 = rearrange(x2, 'b c w h d  -> b (w h d) c', b=b).contiguous()

        
        return x, x1, x2, ids_keep

    def reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x
def rand_bool(*size):
    return torch.randint(2, size) == torch.randint(2, size)

if __name__ == '__main__':

    model = TABS()
    device = torch.device('mps')

    test1 = torch.rand([1,6,336,224,128])
    test3 = torch.rand([1,1,336,224,48])
    test4 = torch.rand([1,1,256,128,32])
    test2 = rand_bool(1, 308)
    #print(test2)
    #test2 = torch.rand([1,6,336,224,64])

    # test = test.to(device)
    # model = model.to(device)

    with torch.no_grad():
        out = model(test1, test4, test3, None)
    print(out[0].shape)
