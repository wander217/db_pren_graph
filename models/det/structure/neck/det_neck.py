import torch
from torch import nn,Tensor

class DBNeck(nn.Module):
    def __init__(self,data_point:tuple,exp:int,bias:bool=False):
        super().__init__()

        # Tầng xử lý dữ liệu từ backbone
        assert len(data_point) >= 4, len(data_point)
        self.in5:nn.Module = nn.Conv2d(data_point[-1],exp,kernel_size=1,bias=bias)
        self.in4:nn.Module = nn.Conv2d(data_point[-2],exp,kernel_size=1,bias=bias)
        self.in3:nn.Module = nn.Conv2d(data_point[-3],exp,kernel_size=1,bias=bias)
        self.in2:nn.Module = nn.Conv2d(data_point[-4],exp,kernel_size=1,bias=bias)

        # Tầng upsampling
        self.up5:nn.Module = nn.Upsample(scale_factor=2)
        self.up4:nn.Module = nn.Upsample(scale_factor=2)
        self.up3:nn.Module = nn.Upsample(scale_factor=2)

        # Tần tạo ra output
        exp_output:int = exp // 4
        self.out5:nn.Module = nn.Sequential(
            nn.Conv2d(exp,exp_output,kernel_size=3,padding=1,bias=bias),
            nn.Upsample(scale_factor=8)
        )
        self.out4:nn.Module = nn.Sequential(
            nn.Conv2d(exp,exp_output,kernel_size=3,padding=1,bias=bias),
            nn.Upsample(scale_factor=4)
        )
        self.out3:nn.Module = nn.Sequential(
            nn.Conv2d(exp,exp_output,kernel_size=3,padding=1,bias=bias),
            nn.Upsample(scale_factor=2)
        )
        self.out2:nn.Module = nn.Sequential(
            nn.Conv2d(exp,exp_output,kernel_size=3,padding=1,bias=bias)
        )

        # Cài đặt giá trị ban đầu
        self.in5.apply(self.weight_init)
        self.in4.apply(self.weight_init)
        self.in3.apply(self.weight_init)
        self.in2.apply(self.weight_init)

        self.out5.apply(self.weight_init)
        self.out4.apply(self.weight_init)
        self.out3.apply(self.weight_init)
        self.out2.apply(self.weight_init)

    def weight_init(self,module):
        class_name = module.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.kaiming_normal_(module.weight.data)
        elif class_name.find("BatchNorm") != -1:
            module.weight.data.fill_(1.)
            module.bias.data.fill_(1e-4)

    def forward(self,x:list)->Tensor:
        assert len(x) == 4, len(x)

        # Xử lý input
        fin5:Tensor = self.in5(x[3])
        fin4:Tensor = self.in4(x[2])
        fin3:Tensor = self.in3(x[1])
        fin2:Tensor = self.in2(x[0])

        # Xử lý upsampling
        fup4:Tensor = self.up5(fin5) + fin4
        fup3:Tensor = self.up4(fup4) + fin3
        fup2:Tensor = self.up3(fup3) + fin2

        # Xử lý output
        fout5:Tensor = self.out5(fin5)
        fout4:Tensor = self.out4(fup4)
        fout3:Tensor = self.out3(fup3)
        fout2:Tensor = self.out2(fup2)

        # Concat lại
        fusion:Tensor = torch.cat([fout5,fout4,fout3,fout2],1)
        return fusion