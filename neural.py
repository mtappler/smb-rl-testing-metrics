import copy
from torch import nn
from torch.nn.functional import relu

from neuron_coverage import CoverageInfo

class MarioNet(nn.Module):
    """
    The class implementing/storing the policy network of the RL agent. The learning logic is implemented in agent.py,
    this class derives from PyTorch neural network modules class, so that it can be used for learning in a convenient
    way.
    The module consists of a target and an online network, both with the same structure, which are sequential
    compositions of the convolutional layers and two linear layers that approximate the Q function for deep RL.
    """
  
    def __init__(self, input_dim, output_dim,nc_threshold=0.0):
        """
        Constructur instantiating the online and the target network.

        Args:
            input_dim: number of actions, size of the output
            output_dim: size of the observations, the input size of the neural network
        """
        super().__init__()
        # observations are tensors of the shape [c=4, h=84, w=84]
        c, h, w = input_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # convolutional part as sequence of three conv. layers
        self.online_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.output_dim = output_dim
        # linear part from 3136 to 512 to #actions
        self.online_linear = nn.Sequential(nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        self.nc_threshold = nc_threshold

        # simply copy the online network to get the initial target network
        self.target_conv = copy.deepcopy(self.online_conv)
        self.target_linear = copy.deepcopy(self.online_linear)
        
        # Q_target parameters are frozen, as we update the target network by copying the online network
        for p in self.target_conv.parameters():
            p.requires_grad = False
        for p in self.target_linear.parameters():
            p.requires_grad = False
        
    def check_coverage(self, state):
        layer1 = self.online_conv[0]
        layer2 = self.online_conv[2]
        layer3 = self.online_conv[4]
        layer4 = self.online_linear[0]
        layer5 = self.online_linear[2]
        act1 = layer1(state)
        act2 = layer2(relu(act1))
        act3 = layer3(relu(act2)).flatten()
        act4 = layer4(relu(act3))        
        act5 = layer5(relu(act4))
        return CoverageInfo([act1.flatten() > self.nc_threshold,
                             act2.flatten() > self.nc_threshold,
                             act3 > self.nc_threshold,
                             act4 > self.nc_threshold,
                             act5 > self.nc_threshold])

    def importance_value(self,state):
        state = state.unsqueeze(0) 
        output = self.forward(state,"online")
        max_q = output.max(axis = 1).values.item()
        min_q = output.min(axis = 1).values.item()

        return (max_q - min_q)/abs(max_q)
    
    def reset_linear(self,use_cuda):
        """
        Resets the linear parts of the networks by newly instantiating them, thus initializing their weights.
        Args:
            use_cuda: True if cuda i.e. the GPU shall be used

        Returns: None

        """
        self.online_linear = nn.Sequential(nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )        
        self.target_linear = copy.deepcopy(self.online_linear)
        # send to the GPU
        if use_cuda:
            self.online_linear = self.online_linear.to(device='cuda')
            self.target_linear = self.target_linear.to(device='cuda')

    def forward(self, input, model):
        """
        Method for applying a neural network computation on an input, i.e., computing Q-values for an observation/a
        state.
        Args:
            input: a state/observation, i.e., stacked downscaled images
            model: string indicating the network to be used, either "online" or "target"

        Returns: result of the neural network computation (Q-values)

        """
        conv_input = input
        if model == 'online':
            conv_res = self.online_conv(conv_input)
            linear_input = conv_res
            return self.online_linear(linear_input)
        elif model == 'target':
            conv_res = self.target_conv(conv_input)
            linear_input = conv_res
            return self.target_linear(linear_input)
