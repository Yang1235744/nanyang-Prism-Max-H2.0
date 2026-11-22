# nanyang-Prism-Max-H2.0 —— 从零自研的图像/视频生成底座
__version__ = "2.0.0a"

from prism.models.text_embed import ByteTextEncoder, encode_text, decode_ids, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID
from prism.models.self_vae import SelfVAE, vae_loss
from prism.models.self_dit import SelfDiT, patchify_3d, unpatchify_3d, timestep_embedding
from prism.diffusion.flow import rectified_flow_loss, EDCISampler
from prism.training.vae_trainer import VAETwoStageTrainer
from prism.training.dit_trainer import DiTRectifiedFlowTrainer
