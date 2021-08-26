# truncate a model to create a feature extractor
def build_feature_extractor(model):
   fc_channels = getattr(model, 'fc').in_features if hasattr(model, 'fc') else 0
   if 'resnet' in  model.network_name:
        # remove the last two layers
        del model.fc
        del model.dropout
        del model.avgpool

   return model, fc_channels
