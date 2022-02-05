"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import ViTForImageClassification
import torchvision.models as models

class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.num_inputs = num_inputs
        self.conv1 = nn.Conv2d(12, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        # self.linear = nn.Linear(151296, 512)
        # self.linear = nn.Linear(1000, 512)
        self.flatten = nn.Flatten()
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vgg11 = models.vgg11() 

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # print(self.num_inputs)
        # exit()
        # print(x.shape)
        # exit()

        # x = torch.reshape(x, (-1, 3, 84, 84))
        # x = self.vgg11(x)

        # x = x[:,:3,:,:]
        # x = F.interpolate(x, (224, 224))
        # x = self.model(x, output_hidden_states=True)
        # x = x["hidden_states"][-1]
        # print(x["hidden_states"])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.linear(x)
        return self.actor_linear(x), self.critic_linear(x)
