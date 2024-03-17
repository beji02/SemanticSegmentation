import torch
import torchvision
from typing import List


class MyUNet(torch.nn.Module):
    def __init__(self, num_classes, encoder_channels, decoder_channels):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.finalFC = torch.nn.Sequential(
            torch.nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )

    def forward(self, x):
        image_size = x.shape[-1]
        x, encoder_output_list = self.encoder(x)
        x = self.decoder(x, encoder_output_list)
        x = self.finalFC(x)
        x = torch.nn.functional.interpolate(x, size=(image_size, image_size))
        return x


class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                encoder_block(channels[i], channels[i+1])
            )
            for i in range(len(channels)-1)
        )
        self.pooling = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        layer_output_list = []
        first = True
        for block in self.blocks:
            if not first:
                x = self.pooling(x)
            else:
                first = False
            x = block(x)
            layer_output_list.append(x)
        
        layer_output_list.pop()
        
        return x, layer_output_list
    

class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder_blocks = torch.nn.ModuleList(
            decoder_block(channels[i], channels[i+1])
            for i in range(len(channels) - 1)
        )
        self.encoder_blocks = torch.nn.ModuleList(
            encoder_block(channels[i], channels[i+1]) # TODO not ok
            for i in range(len(channels) - 1)
        )
    
    def forward(self, x, skips: List[torch.Tensor]):
        skips = skips[::-1]
        for i, (decoder_block, encoder_block) in enumerate(zip(self.decoder_blocks, self.encoder_blocks)):
            skip = skips[i]
            x = decoder_block(x)
            skip = torch.nn.functional.interpolate(skip, size=x.shape[-1], mode='nearest')
            x = torch.cat([x, skip], 1)
            x = encoder_block(x)
        
        return x
 

def encoder_block(channels_in, channels_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, 3),
        torch.nn.ReLU(),
    )


def decoder_block(channels_in, channels_out):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(channels_in, channels_out, 2, 2),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU()
    )

    
if __name__ == "__main__":
    #test encoder
    testX = torch.rand(15, 1, 572, 572)
    encoder_channels = [1, 64, 128, 256, 512, 1024]

    x, output_list = Encoder(encoder_channels)(testX)
    for output in output_list:
        print(output.shape)
    print(x.shape)

    # test blocks
    testX = torch.rand(15, 1024, 28, 28)
    testSkip = torch.rand(15, 512, 64, 64)

    x = decoder_block(1024, 512)(testX)
    skip = torchvision.transforms.CenterCrop(x.shape[2])(testSkip)
    print(skip.shape)
    x = torch.cat([x, skip], 1)

    x = encoder_block(1024, 512)(x)
    print(x.shape)

    # test decoder
    testX = torch.rand(15, 1024, 28, 28)
    testSkip = torch.rand(15, 512, 64, 64)
    decoder_channels = [1024, 512, 256, 128, 64]

    x = Decoder(decoder_channels)(testX, output_list)
    print(x.shape)

    # test Unet
    testX = torch.rand(15, 1, 250, 250)
    num_classes = 2
    encoder_channels = [1, 64, 128, 256, 512, 1024]
    decoder_channels = [1024, 512, 256, 128, 64]

    myUNet = MyUNet(num_classes, encoder_channels, decoder_channels)
    output = myUNet(testX)

    print(output.shape)


