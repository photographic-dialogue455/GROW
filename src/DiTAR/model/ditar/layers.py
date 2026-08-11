import torch.nn as nn

class Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        w_init_gain: str = 'linear',
        activation = None,
        **kwargs
    ):
        super(Linear, self).__init__(
            in_channels,
            out_channels,
            bias=bias
        )

        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = out_channels
        if w_init_gain is not None:
            if isinstance(w_init_gain, str):
                gain = nn.init.calculate_gain(w_init_gain)
            else:
                gain = w_init_gain
            nn.init.xavier_uniform_(
                    self.weight, gain=gain)
        if bias:
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x, **kwargs):
        return self.activation(super(Linear, self).forward(x))