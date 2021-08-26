import torch
from torch import nn
from functools import partial
import torch.utils.model_zoo as model_zoo
#from .utils import load_state_dict_from_url
from .temporal_modeling import temporal_modeling_module
from model_zoo.inflate_from_2d_model import convert_rgb_model_to_group
from model_zoo.eca_module import eca_layer
from inspect import signature
from .mobilenet import MobileNetV2
from model_zoo.feature_extractor import build_feature_extractor

__all__ = ['SSTMobileNetV2', 'SST_mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class SSTMobileNetV2(nn.Module):
    def __init__(self,
                 num_frames,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None, 
                 dropout=0.5,
                 without_t_stride=False, 
                 temporal_module=None,
                 pooling_method='max'
                ):
        super(SSTMobileNetV2, self).__init__()

        self.num_frames = num_frames
        self.temporal_module = temporal_module
        self.width_mult = width_mult
        self.add_norm = True

        self.tsn = MobileNetV2(num_frames=num_frames, 
                        num_classes=num_classes,
                        width_mult=width_mult,
                        inverted_residual_setting = inverted_residual_setting,
                        round_nearest = round_nearest,
                        block = block,
                        dropout = dropout,
                        without_t_stride = without_t_stride,
                        temporal_module=None)
        self.tsn, tsn_fc_channels = build_feature_extractor(self.tsn)

        self.tam = MobileNetV2(num_frames=num_frames,
                        num_classes=num_classes,
                        width_mult=width_mult,
                        inverted_residual_setting = inverted_residual_setting,
                        round_nearest = round_nearest,
                        block = block,
                        dropout = dropout,
                        without_t_stride = without_t_stride,
                        temporal_module=temporal_module)

        self.tam, tam_fc_channels = build_feature_extractor(self.tam)

        self.tsn_attention = eca_layer(1280, normalize=self.add_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(tsn_fc_channels + tam_fc_channels, num_classes)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        batch_size, c_t, h, w = x.shape
        # NXFxWxH
        out1 = self.tsn.extract_feature(x)
        out1 = self.avgpool(out1)
        out1 = self.tsn_attention(out1)
        if self.add_norm:
            out1 = out1.view(batch_size, self.num_frames, -1, 1, 1)
            out1 = torch.sum(out1, dim=1, keepdim=True)
            #expand
            out1 = out1.repeat(1, self.num_frames, 1, 1, 1)
            out1 = out1.view(batch_size * self.num_frames, -1, 1, 1)
             
        # NXFxWxH
        out2 = self.tam.extract_feature(x)
        out2 = self.avgpool(out2)
        #concatenation features
        out = torch.cat((out1, out2), 1)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)

        n_t, c = out.shape
        out = out.view(batch_size, -1, c)

        # average the prediction from all frames
        out = torch.mean(out, dim=1)

        #x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        #x = self.classifier(x)
        return out
        
    def forward(self, x):
        return self._forward_impl(x)

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            name += "{}-".format(temporal_module)
        name += 'SST-mobilenetV2-{}'.format(int(self.width_mult*100))
        #print (name)
        return name

def SST_mobilenet_v2(width_mult, num_classes, without_t_stride, groups, temporal_module_name,
           dw_conv, blending_frames, blending_method, dropout, pooling_method, input_channels, imagenet_pretrained=True, **kwargs):

    temporal_module = partial(temporal_modeling_module, name=temporal_module_name,
                              dw_conv=dw_conv,
                              blending_frames=blending_frames,
                              blending_method=blending_method) if temporal_module_name is not None \
        else None

    model = SSTMobileNetV2(num_frames=groups, 
                        num_classes=num_classes,
                        width_mult=width_mult,
                        inverted_residual_setting = None,
                        round_nearest = 8,
                        block = None,
                        dropout = dropout,
                        without_t_stride = without_t_stride,
                        temporal_module=temporal_module,
                        pooling_method = pooling_method)

#    for key, value in model.state_dict().items():
#        if key == 'features.1.conv.0.0.weight':
#            print (key, value.shape)

    if imagenet_pretrained:
        #state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], map_location='cpu')
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'], map_location='cpu')
        model.tsn.load_state_dict(state_dict, strict=False)
        model.tam.load_state_dict(state_dict, strict=False)
    
   # for name, param in model.named_parameters():
   #     print (name, param.data.shape)
    #for key, value in state_dict.items():
    #    print (key, value)
    return model
