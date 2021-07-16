import torch
import torch.nn as nn

class GlobalNet(nn.Module):
    def __init__(self, n_channels, n_classes, ngf):
        # Todo: FC muss runtergebrochen und danach wieder zu [batch, width, height, features] zusammengefügt werden
        super().__init__()
        self.encoder1 = Encoder1(n_channels, ngf)
        self.encoder2 = Encoder(ngf, ngf*2)
        self.encoder3 = Encoder(ngf*2, ngf*4)
        self.encoder4 = Encoder(ngf*4, ngf*8)
        self.encoder5 = Encoder(ngf*8, ngf*8)
        self.encoder6 = Encoder(ngf*8, ngf*8)
        self.encoder7 = Encoder(ngf*8, ngf*8)
        self.encoder8 = Encoder(ngf*8, ngf*8, False)
        self.decoder8 = Decoder(ngf*8, ngf*16)
        self.decoder7 = Decoder(ngf*16, ngf*16)
        self.decoder6 = Decoder(ngf*16, ngf*16)
        self.decoder5 = Decoder(ngf*16, ngf*16)
        self.decoder4 = Decoder(ngf*16, ngf*8)
        self.decoder3 = Decoder(ngf*8, ngf*4)
        self.decoder2 = Decoder(ngf*4, ngf*2)
        self.decoder1 = Decoder1(ngf*2, n_classes)

    def forward(self, x):

class Encoder1(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.conv = nn.Conv2d(input_features, output_features, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1))
        self.fcselu = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.SELU()
        )
    def forward(self, inp):
        output = self.conv(inp)
        mean = torch.mean(inp, dim=(2, 3), keepdim=True)
        gl_output = self.fcselu(mean)

class Encoder(nn.Module):
    def __init__(self, input_features, output_features, inNorm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(input_features, output_features, kernel_size=(1,1), stride=(2,2), padding=(1,1)),
        )
        self.inNorm = nn.InstanceNorm2d(output_features) if inNorm else nn.Identity()

        self.fc = nn.Linear(input_features, output_features)
        self.fcselu = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.SELU()
        )

    def forward(self, inp, gl_inp):
        conved = self.conv(inp)
        output = self.inNorm(conved) + self.fc(gl_inp)

        #global track
        mean = torch.mean(conved, dim=(2, 3), keepdim=True)
        concatenated = torch.cat(gl_inp, mean, dim=-1)
        gl_output = self.fcselu(concatenated)
        return output, gl_output

class Decoder(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(input_features, output_features, kernel_size=(1,1), stride=(1,1)),
            nn.Conv2d(output_features, output_features, kernel_size=(1,1), stride=(1,1)),
        )
        self.inNorm = nn.InstanceNorm2d(output_features)
        self.dropout = nn.Dropout2d(0.5)

        self.fc = nn.Linear(input_features, output_features)
        self.fcselu = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.SELU()
        )

    def forward(self, inp, gl_inp, skip):
        if skip is not None: inp = torch.cat(inp, skip, dim=3)
        in_height, in_width = inp.size()[1], inp.size()[2]
        inp.resize((inp.size()[0], in_height*2, in_width*2, inp.size()[3]))
        conved = self.conv(inp)
        output = self.inNorm(conved) + self.fc(gl_inp)
        output = self.dropout(output)

        #global track
        mean = torch.mean(conved, dim=(2, 3), keepdim=True)
        concatenated = torch.cat(gl_inp, mean)
        gl_output = self.fcselu(concatenated)
        return output, gl_output

class Decoder1:
    def __init__(self, input_features, output_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(input_features, output_features, kernel_size=(1,1), stride=(2,2), padding=(1,1)),
        )

        self.tanh = nn.Tanh()

    def forward(self, inp, gl_inp, skip):
        inp = torch.cat(inp, skip, dim=3)
        conved = self.conv(inp)

        output = conved + self.fc(gl_inp)
        output = self.tanh(output)
        return output
