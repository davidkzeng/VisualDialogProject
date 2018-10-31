from .lf import LateFusionEncoder
from .hre2 import HierarchicalRecurrentEncoder

name_enc_map = {
    'lf-ques-im-hist': {
    	'encoder': LateFusionEncoder,
    	'params' : {
            'concat_history' : True
    	}
    }, 
    'hre-ques-im-hist': {
    	'encoder': HierarchicalRecurrentEncoder, 
    	'params' : {
            'concat_history' : False
    	}
    }
}

def Encoder(model_args):
    return name_enc_map[model_args.encoder]['encoder'](model_args)

def EncoderParams(model_args):
	return name_enc_map[model_args.encoder]['params']