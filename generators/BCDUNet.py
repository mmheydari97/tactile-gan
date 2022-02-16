import torch 
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    Basic CLSTM cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, b, h, w):
        return (torch.zeros(b, self.hidden_dim, h, w).cuda(),
                torch.zeros(b, self.hidden_dim, h, w).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list  # , last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBLSTM(nn.Module):
    # Constructor
    def __init__(self, in_channels, hidden_channels,
                 kernel_size, num_layers, bias=True, batch_first=True):

        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(in_channels, hidden_channels//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias)
        self.reverse_net = ConvLSTM(in_channels, hidden_channels//2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias)
        
    def forward(self, x):
        """
        x = B T C H W tensors.
        """
        
        y_out_fwd = self.forward_net(x)
        reversed_idx = list(reversed(range(x.shape[1])))
        y_out_rev = self.reverse_net(x[:, reversed_idx, ...])
        
        y_out_fwd = y_out_fwd[-1] # outputs of last CLSTM layer = B, T, C, H, W
        y_out_rev = y_out_rev[-1] # outputs of last CLSTM layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...] # reverse temporal outputs.
        b,t,c,h,w = y_out_fwd.shape
        y_out_fwd = torch.reshape(y_out_fwd, (b, t*c, h, w))
        y_out_rev = torch.reshape(y_out_rev, (b, t*c, h, w))

        ycat = torch.cat((torch.squeeze(y_out_fwd, 1), torch.squeeze(y_out_rev, 1)), dim=1)
        
        return ycat


class BCDUNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, num_filter=64, norm='instance', bidirectional=False):
        super(BCDUNet, self).__init__()
        self.num_filter = num_filter
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()
        self.dropout = nn.Dropout(0.5).cuda()

        self.conv1_0 = nn.Conv2d(input_dim, num_filter, kernel_size=3, stride=1, padding=1).cuda()
        self.conv1_1 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2_0 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2_1 = nn.Conv2d(num_filter*2, num_filter*2, kernel_size=3, stride=1, padding=1).cuda()
        self.conv3_0 = nn.Conv2d(num_filter*2, num_filter*4, kernel_size=3, stride=1, padding=1).cuda()
        self.conv3_1 = nn.Conv2d(num_filter*4, num_filter*4, kernel_size=3, stride=1, padding=1).cuda()
        self.conv4_0 = nn.Conv2d(num_filter*4, num_filter*8, kernel_size=3, stride=1, padding=1).cuda()
        self.conv4_1 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1).cuda()
        self.conv4_2 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1).cuda()
        self.conv4_3 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1).cuda()
        self.conv4_4 = nn.Conv2d(num_filter*16, num_filter*8, kernel_size=3, stride=1, padding=1).cuda()
        self.conv4_5 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1).cuda()
        
        self.conv6_0 = nn.Conv2d(num_filter*2, num_filter*4, kernel_size=3, stride=1, padding=1).cuda()
        self.conv6_1 = nn.Conv2d(num_filter*4, num_filter*4, kernel_size=3, stride=1, padding=1).cuda()

        self.conv7_0 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, stride=1, padding=1).cuda()
        self.conv7_1 = nn.Conv2d(num_filter*2, num_filter*2, kernel_size=3, stride=1, padding=1).cuda()

        self.conv8_0 = nn.Conv2d(num_filter//2, num_filter, kernel_size=3, stride=1, padding=1).cuda()
        self.conv8_1 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1).cuda()
        self.conv8_2 = nn.Conv2d(num_filter, num_filter//2, kernel_size=3, stride=1, padding=1).cuda()

        self.conv9_0 = nn.Conv2d(num_filter//2, output_dim, kernel_size=1, stride=1).cuda()

        self.convt1 = nn.ConvTranspose2d(num_filter*8, num_filter*4, kernel_size=2, stride=2, padding=0).cuda()
        self.bn1 = nn.BatchNorm2d(num_filter*4).cuda()
        self.convt2 = nn.ConvTranspose2d(num_filter*4, num_filter*2, kernel_size=2, stride=2, padding=0).cuda()
        self.bn2 = nn.BatchNorm2d(num_filter*2).cuda()
        self.convt3 = nn.ConvTranspose2d(num_filter*2, num_filter, kernel_size=2, stride=2, padding=0).cuda()
        self.bn3 = nn.BatchNorm2d(num_filter).cuda()

        if bidirectional:

            self.clstm1 = ConvBLSTM(in_channels=num_filter*4, hidden_channels=num_filter, kernel_size=(3, 3), num_layers=1).cuda()
            self.clstm2 = ConvBLSTM(in_channels=num_filter*2, hidden_channels=num_filter//2, kernel_size=(3, 3), num_layers=1).cuda()
            self.clstm3 = ConvBLSTM(in_channels=num_filter, hidden_channels=num_filter//4, kernel_size=(3, 3), num_layers=1).cuda()
        else:
            self.clstm1 = ConvLSTM(in_channels=num_filter*4, hidden_channels=num_filter, kernel_size=(3, 3), num_layers=1).cuda()
            self.clstm2 = ConvLSTM(in_channels=num_filter*2, hidden_channels=num_filter//2, kernel_size=(3, 3), num_layers=1).cuda()
            self.clstm3 = ConvLSTM(in_channels=num_filter, hidden_channels=num_filter//4, kernel_size=(3, 3), num_layers=1).cuda()

            
    def forward(self, x):
        N = x.size()[-2]
        conv1 = self.conv1_0(x)
        conv1 = self.conv1_1(conv1)
        pool1 = self.maxpool(conv1)
        conv2 = self.conv2_0(pool1)
        conv2 = self.conv2_1(conv2)
        pool2 = self.maxpool(conv2)
        conv3 = self.conv3_0(pool2)
        conv3 = self.conv3_1(conv3)
        drop3 = self.dropout(conv3)
        pool3 = self.maxpool(conv3)
        # D1
        conv4 = self.conv4_0(pool3)
        conv4_1 = self.conv4_1(conv4)
        drop4_1 = self.dropout(conv4_1)
        # D2
        conv4_2 = self.conv4_2(drop4_1)
        conv4_2 = self.conv4_3(conv4_2)
        conv4_2 = self.dropout(conv4_2)
        # D3
        merge_dense = torch.cat((conv4_2, drop4_1), 1)
        conv4_3 = self.conv4_4(merge_dense)
        conv4_3 = self.conv4_5(conv4_3)
        drop4_3 = self.dropout(conv4_3)
 
        up6 = self.convt1(drop4_3)
        up6 = self.bn1(up6)
        up6 = nn.ReLU()(up6)

        x1 = drop3.view(-1,1,self.num_filter*4,N//4,N//4)
        x2 = up6.view(-1,1,self.num_filter*4,N//4,N//4)

        merge6 = torch.cat((x1, x2), 1)
        merge6 = self.clstm1(merge6)
        
        conv6 = self.conv6_0(merge6)
        conv6 = self.conv6_1(conv6)

        up7 = self.convt2(conv6)
        up7 = self.bn2(up7)
        up7 = nn.ReLU()(up7)

        x1 = conv2.view(-1,1,self.num_filter*2,N//2,N//2)
        x2 = up7.view(-1,1,self.num_filter*2,N//2,N//2)
        merge7 = torch.cat((x1, x2), 1)
        merge7 = self.clstm2(merge7)

        conv7 = self.conv7_0(merge7)
        conv7 = self.conv7_1(conv7)

        up8 = self.convt3(conv7)
        up8 = self.bn3(up8)
        up8 = nn.ReLU()(up8)

        x1 = conv1.view(-1,1,self.num_filter,N,N)
        x2 = up8.view(-1,1,self.num_filter,N,N)
        merge8 = torch.cat((x1, x2), 1)
        merge8 = self.clstm3(merge8)

        conv8 = self.conv8_0(merge8)
        conv8 = self.conv8_1(conv8)
        conv8 = self.conv8_2(conv8)

        conv9 = self.conv9_0(conv8)

        return conv9
