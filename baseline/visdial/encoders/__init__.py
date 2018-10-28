from .lf import LateFusionEncoder
from .hre2 import HierarchicalRecurrentEncoder

def Encoder(model_args):
    name_enc_map = {
        'lf-ques-im-hist': LateFusionEncoder,
        'hre-ques-im-hist': HierarchicalRecurrentEncoder
    }
    return name_enc_map[model_args.encoder](model_args)

