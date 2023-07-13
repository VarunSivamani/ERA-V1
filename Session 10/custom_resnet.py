import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dropout):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            Custom_Layer(  3,  64, layers=0, dropout=dropout, maxpool=False),
            Custom_Layer( 64, 128, layers=2, dropout=dropout, maxpool=True),
            Custom_Layer(128, 256, layers=0, dropout=dropout, maxpool=True),
            Custom_Layer(256, 512, layers=2, dropout=dropout, maxpool=True),
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x)
    

class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, dropout, bias=False, stride=1, padding=1, maxpool=False):
        super(ConvLayer, self).__init__()

        block = list()
        block.append(nn.Conv2d(input_c, output_c, 3, bias=bias, stride=stride, padding=padding, padding_mode='replicate'))
        if maxpool:
            block.append(nn.MaxPool2d(2, 2))
        block.append(nn.BatchNorm2d(output_c))
        block.append(nn.ReLU())
        block.append(nn.Dropout(dropout))

        self.blocks = nn.Sequential(*block)

    def forward(self, x):
        return self.blocks(x)


class Custom_Layer(nn.Module):
    def __init__(self, input_c, output_c, dropout, maxpool=True, layers=2):
        super(Custom_Layer, self).__init__()

        self.pool_block = ConvLayer(input_c, output_c, dropout=dropout, maxpool=maxpool)
        self.dropout = dropout
        self.residual_block = None

        if layers > 0:
            layer = list()
            for i in range(0, layers):
                layer.append(ConvLayer(output_c, output_c, dropout=dropout, maxpool=False))

            self.residual_block = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pool_block(x)

        if self.residual_block is not None:
            y = x
            x = self.residual_block(x)
            x = x + y
        return x