import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils.PositionalEmbeddings import PosGrid
import einops

# This implementation is based on the original Impelmentation of MAE:
# https://github.com/facebookresearch/mae


class MaskedAutoEncoder(nn.Module):
    
    def __init__(self, input_shape = (1, 64, 64), patch_size = 16,
                encoder_embedding_dim = 768, encoder_depth = 12, encoder_num_heads = 12,
                decoder_embedding_dim = 512, decoder_depth = 8, decoder_num_heads = 8,
                mlp_ratio = 4, num_classes = 3, norm_pix_loss = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.C, self.H, self.W = input_shape
        self.patch_size = patch_size
        self.num_patches = (self.H // patch_size) * (self.W // patch_size) * self.C
        self.encoder_embed_dim = encoder_embedding_dim
        self.decoder_embed_dim = decoder_embedding_dim
        self.num_classes = num_classes
        self.norm_pix_loss = norm_pix_loss

        assert self.H % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert self.W == self.H , 'Image must be square.'
        assert self.num_patches > 0, 'Patches must be greater than 0.'
        
        # General Portion:------------------------------------------------------
        self.patch_embed_func = PatchEmbed( 
                                        img_size=self.H, patch_size=patch_size,
                                        in_chans=self.C, embed_dim=encoder_embedding_dim,
                            )
        self.pos_embed = nn.Parameter(
            PosGrid(
                (self.H // patch_size, self.W // patch_size),
                encoder_embedding_dim, cls_token=True).unsqueeze(0),
            requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embedding_dim))
        self.masking_token = nn.Parameter(torch.randn(1, 1, decoder_embedding_dim))
        

        # Encoder Portion:------------------------------------------------------
        self.encoder_blocks = nn.ModuleList([
            Block(
                dim=encoder_embedding_dim,
                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, proj_drop=0.1,
                attn_drop=0.1, proj_bias=True, qkv_bias=True, qk_norm=True,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embedding_dim)
        
        
        # Decoder Portion:-------------------------------------------------------
        self.decoder_embed_func = nn.Sequential(
            nn.Linear(encoder_embedding_dim, decoder_embedding_dim),
        )
        self.decoder_pos_embed = nn.Parameter(
            PosGrid(
                (self.H // patch_size, self.W // patch_size),
                decoder_embedding_dim, cls_token=False
            ).unsqueeze(0),
            requires_grad=True)
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embedding_dim, num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=True, qk_norm=True, proj_bias=True,
                attn_drop=0.1, proj_drop=0.1
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embedding_dim)
        self.decoder_head = nn.Sequential(
            nn.Linear(decoder_embedding_dim, self.C * patch_size ** 2),
        )

        # Proper initialization of weights:---------------------------------------
        # nn.init.trunc_normal_(self.pos_embed, std=0.5)
        # nn.init.trunc_normal_(self.decoder_pos_embed, std=0.2)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.masking_token, std=0.02)
        w = self.patch_embed_func.proj.weight
        nn.init.xavier_uniform_(w)
        self.apply(self.initialize_weights)


    def initialize_weights(self, m):
        """ 
            This applies the Xavier initialization to every layer in the model.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, x):
        """ 
            This function takes an image tensor and returns patches as tensors.
            Args:
                x: Image tensor of shape (B, C, H, W)
            Returns:
                patches: Tensor of shape (B, (H*W)/(p**2), C*p*p)
        """
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = einops.rearrange(x, 'b c h p w q -> b h w c p q').contiguous()
        x = x.reshape(B, -1, C * p * p)
        return x

    def unpatchify(self, x):
        """ 
            This function takes patches and returns the original image.
            Args:
                x: Tensor of shape (B, (H*W)/(p**2), C*p*p)
            Returns:
                image: Image tensor of shape (B, C, H, W)
        """
        p = self.patch_size
        B, N, L = x.shape
        C = self.C
        H = W = int(self.H)
        x = x.reshape(B, H // p, W // p, C, p, p)
        x = einops.rearrange(x, 'b h w c p q -> b c h p w q')
        x = x.reshape(B, C, H, W)
        return x

    
    def random_masking(self, x, masking_ratio=0.75):
        """ 
            Creates a random mask for the patches.
            Args:
                x: Tensor of shape (B, (H*W)/(p**2), C*p*p)
            Returns:
                - x_masked: Tensor of shape (B, num_unmasked, C*p*p) \\
                - revert_indices: Tensor of shape (B, num_unmasked) for reconstructing the patches \\
                - mask: Tensor of shape (B, (H*W)/(p**2))
        """
        B,L,D = x.shape
        len_keep = L - int(L * masking_ratio)
        
        noise = torch.rand(B, L).to(x.device)
        indices = noise.argsort(dim=1)
        revert_indices = indices.argsort(dim=1)
        indices_keep = indices[:, :len_keep]
        # print(x.device, indices_keep.device)
        x_masked = torch.gather(x, 1, indices_keep.unsqueeze(-1).expand(B, len_keep, D))
        mask = torch.ones(B, L).to(x.device)
        mask.scatter_(1, indices_keep, 0)
        return x_masked, revert_indices, mask

    def forward_encoder(self, x, masking_ratio=0.75, mode = "reconstruction"):
        """ 
            This function passes the images through the encoder.
            Args:
                x: Tensor of shape (B, C, H, W)
            Returns:
                x: Tensor of shape (B, (H*W)/(p**2) + 1, C*p*p)
                mask: Tensor of shape (B, (H*W)/(p**2))
                revert_index: Tensor of shape (B, (H*W)/(p**2))
        """
        assert mode in ["reconstruction", "classification"], "Mode must be either 'reconstruction' or 'classification'"
        # print(x.shape)
        patches = self.patch_embed_func(x)
        # print(patches.shape, self.pos_embed.shape)
        patches += self.pos_embed[:, 1:, :] * 4
        x, revert_index, mask = self.random_masking(patches, masking_ratio)
        if mode == "classification":
            cls_tokens = self.cls_token + (self.pos_embed[:, 0, :]*10)
            cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder_norm(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x, mask, revert_index

        
    def forward_decoder(self, x, revert_indx, mode="reconstruction"):
        """ 
            This function passes the patches through the decoder.
            
            Args:
                x: Tensor of shape (B, (H*W)/(p**2), C*p*p)
                revert_indx: Tensor of shape (B, (H*W)/(p**2))
                
            Returns:
                x: Tensor of shape (B, C, H, W)
        """

        assert mode in ["reconstruction", "classification"], "Mode must be either 'reconstruction' or 'classification'"
        if mode == "classification":
            x = x[:, 1:, :]
        x = self.decoder_embed_func(x)

        # Appending the positional embeddings
        mask_tokens = self.masking_token.expand(x.shape[0], self.num_patches - x.shape[1], -1)
        # print(x.shape, mask_tokens.shape)
        x = torch.cat((x, mask_tokens), dim=1)
        x = torch.gather(x, 1,
                revert_indx.unsqueeze(-1).expand(x.shape[0], self.num_patches, x.shape[-1]))
        # x = torch.cat((x[:, :1, :], x_), dim=1)

        # if self.training:
        #    noise = torch.randn_like(self.decoder_pos_embed) * 0.1
        #     x += self.decoder_pos_embed + noise
        # else:
        #     x += self.decoder_pos_embed
        
        x += self.decoder_pos_embed * 4
        x = self.decoder_norm(x)
        for block in self.decoder_blocks:
            x = block(x)
        # x = self.decoder_norm(x)
        x = self.decoder_head(x)
        return x



    def forward_loss(self, imgs, pred, mask):
        """ 
        Args:
        imgs: Tensor of shape (B, C, H, W)
        pred: Tensor of shape (B, C, H, W)
        mask: Tensor of shape (B, (H*W)/(p**2))
        Returns:
        loss: Tensor of shape (1)
        """

        target = self.patchify(imgs)
        if(self.norm_pix_loss):
            mean = target.mean(dim=-1, keepdim=True)
            std = target.var(dim=-1, keepdim=True)
            target = (target - mean) / ( std + 1e-8)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        unmasked_loss = (loss * (1-mask)).sum() / ((1-mask).sum() + 1e-8)

        total_loss = 0.7*loss + 0.3*unmasked_loss
        return total_loss
        
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
