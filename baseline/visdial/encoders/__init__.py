from .lf import LateFusionEncoder
from .hre2 import HierarchicalRecurrentEncoder
from .lf_short_hist import LateFusionShortHistEncoder
# from .tent_att import TentativeAttentionEncoder

name_enc_map = {
    'lf-ques-im-hist': {
    	'encoder': LateFusionEncoder,
    	'params' : {
            'concat_history' : True,
            'partial_concat_history' : False
    	}
    }, 
    'lf-ques-im-hist-short': {
        'encoder' : LateFusionShortHistEncoder,
        'params' : {
            'concat_history' : False,
            'partial_concat_history' : True
        }
    },
    'hre-ques-im-hist': {
    	'encoder': HierarchicalRecurrentEncoder, 
    	'params' : {
            'concat_history' : False,
            'partial_concat_history' : False
    	}
    }
    #'ta-ques-im-hist': {
    #    'encoder' : TentativeAttentionEncoder,
    #    'params' : {
    #        'concat_history' : False,
    #        'partial_concat_history' : False
    #    }
    #}
}

def Encoder(model_args):
    return name_enc_map[model_args.encoder]['encoder'](model_args)

def EncoderParams(model_args):
	return name_enc_map[model_args.encoder]['params']