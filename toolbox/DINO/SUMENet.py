import torch
import torch.nn as nn
import torch.nn.functional as F
from MLPDecoder import DecoderHead
from thop import profile


def _reshape_token_to_map(tok, h, w):
    """[B, N, C] → [B, C, h, w]"""
    return tok.transpose(1, 2).reshape(tok.shape[0], -1, h, w)


class UnderwaterColorCompensationAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = dim // reduction

        # ── 语义细化路径（保留原 Adapter 能力）──
        self.norm = nn.LayerNorm(dim)
        self.semantic_down = nn.Linear(dim, hidden)
        self.semantic_up = nn.Linear(hidden, dim)
        self.semantic_gate = nn.Linear(dim, dim)
        self.color_comp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
        self.freq_attn = nn.Sequential(
            nn.Linear(dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, dim),
            nn.Sigmoid()
        )
        self.register_buffer(
            'attenuation_prior',
            torch.ones(1, 1, dim)
        )
        with torch.no_grad():
            nn.init.constant_(self.attenuation_prior, 0.9)

        self.act = nn.GELU()
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        sem_delta = self.semantic_up(self.act(self.semantic_down(x_n)))
        sem_gate = torch.sigmoid(self.semantic_gate(x_n))
        sem_out = x + sem_gate * sem_delta
        color_gain = self.color_comp(x_n)  # [B, N, C]
        color_out = x * (1.0 + self.alpha * color_gain)
        freq_weight = self.freq_attn(x_n)  # [B, N, C]
        freq_out = x * freq_weight
        return sem_out + self.alpha * color_out + self.alpha * freq_out

class BlockWithAdapter(nn.Module):
    def __init__(self, block: nn.Module, dim: int):
        super().__init__()
        self.block   = block
        self.adapter = UnderwaterColorCompensationAdapter(dim)

    def forward(self, x: torch.Tensor, rop_sincos=None):
        x = self.block(x, rop_sincos)
        x = self.adapter(x)
        return x


class HyperdimensionalEpistemicEncoder(nn.Module):

    def __init__(self, dim: int, hyper_dim: int = 512):
        super().__init__()
        project_dim = 128
        self.channel_proj = nn.Conv2d(dim, project_dim, 1)
        self.spatial_proj = nn.Conv2d(dim, project_dim, 1)
        self.register_buffer("hyper_proj", torch.randn(project_dim, hyper_dim))

    def forward(self, depth_feat: torch.Tensor) -> torch.Tensor:
        B, C, h, w = depth_feat.shape
        ch_feat  = self.channel_proj(depth_feat).mean(dim=[2, 3])
        ch_hyper = ch_feat @ self.hyper_proj
        ch_unc   = torch.sigmoid(-torch.norm(ch_hyper, dim=1, keepdim=True)
                                 ).unsqueeze(-1).unsqueeze(-1)
        sp_feat  = self.spatial_proj(depth_feat).flatten(2).transpose(1, 2)
        sp_hyper = sp_feat @ self.hyper_proj
        sp_unc   = torch.sigmoid(-torch.norm(sp_hyper, dim=-1, keepdim=True)
                                 ).transpose(1, 2).reshape(B, 1, h, w)
        epistemic_unc = (ch_unc + sp_unc) / 2.0
        return epistemic_unc


class EvidentialDepthFeatureExtractor(nn.Module):

    def __init__(self, dim: int, hyper_dim: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim // 4),
            nn.SiLU(),
        )
        self.nig_head = nn.Conv2d(dim // 4, 4, 1)
        self.hae = HyperdimensionalEpistemicEncoder(dim // 4, hyper_dim)

    def forward(self, depth_feat: torch.Tensor):
        feat = self.shared(depth_feat)

        # ── NIG 参数（softplus 约束）──
        raw   = self.nig_head(feat)
        gamma = raw[:, 0:1]
        nu    = F.softplus(raw[:, 1:2]) + 1e-4
        alpha = F.softplus(raw[:, 2:3]) + 1.0 + 1e-4
        beta  = F.softplus(raw[:, 3:4]) + 1e-4

        aleatoric_unc = beta / (alpha - 1.0)
        epistemic_unc = self.hae(feat)

        nig_params = dict(gamma=gamma, nu=nu, alpha=alpha, beta=beta)
        return aleatoric_unc, epistemic_unc, nig_params


class EUGM(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.edfe  = EvidentialDepthFeatureExtractor(dim)

    def forward(self, depth_feat):
        ale_unc, epi_unc, nig_params = self.edfe(depth_feat)
        task_unc = ale_unc + epi_unc * (1.0 - ale_unc)    # [B,1,h,w]

        return task_unc, nig_params


class DualStreamIndependentRouter(nn.Module):

    def __init__(self, dim: int, num_layers: int = 12, topk: int = 4,
                 tau_init: float = 1.0, tau_min: float = 0.1):
        super().__init__()
        self.topk     = topk
        self.tau_min  = tau_min
        self.register_buffer('tau', torch.tensor(tau_init))
        half = dim // 2


        self.rgb_self_proj  = nn.Linear(dim, half)
        self.rgb_cross_proj = nn.Linear(dim, half)
        self.rgb_fc = nn.Sequential(
            nn.LayerNorm(half),
            nn.Linear(half, num_layers)
        )


        self.depth_self_proj  = nn.Linear(dim, half)
        self.depth_cross_proj = nn.Linear(dim, half)
        self.depth_fc = nn.Sequential(
            nn.LayerNorm(half),
            nn.Linear(half, num_layers)
        )

        self._aux_loss_rgb   = None
        self._aux_loss_depth = None

    def anneal_tau(self, factor: float = 0.995):
        new_tau = max(self.tau.item() * factor, self.tau_min)
        self.tau.fill_(new_tau)

    def _route(self, self_feat, cross_feat, fc, store_attr):
        h      = self_feat + cross_feat
        logits = fc(h)

        if self.training:
            gumbel = -torch.log(-torch.log(
                torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            ))
            weights = F.softmax((logits + gumbel) / self.tau, dim=-1)

            mean_load = weights.mean(0)
            aux = (mean_load * torch.log(mean_load + 1e-6)).sum()
            setattr(self, store_attr, aux)
        else:
            weights = F.softmax(logits, dim=-1)

        _, topk_idx = torch.topk(weights, self.topk, dim=-1)   # [B, k]
        return weights, topk_idx

    def forward(self, cls_rgb: torch.Tensor, cls_depth: torch.Tensor):
        r_self  = self.rgb_self_proj(cls_rgb)
        d_self  = self.depth_self_proj(cls_depth)

        r_cross = self.rgb_cross_proj(cls_depth)
        d_cross = self.depth_cross_proj(cls_rgb)

        rgb_weights, rgb_topk = self._route(
            r_self, r_cross, self.rgb_fc, '_aux_loss_rgb'
        )
        depth_weights, depth_topk = self._route(
            d_self, d_cross, self.depth_fc, '_aux_loss_depth'
        )

        return rgb_weights, rgb_topk, depth_weights, depth_topk

class RGBDominantExpert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rgb_enc  = nn.Conv2d(dim, dim, 3, padding=1, groups=dim // 32, bias=False)
        self.depth_gating = nn.Sequential(
            nn.Conv2d(dim + 1, dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(16, dim),
        )

    def forward(self, rgb, depth, uncertainty):
        rgb_feat = self.rgb_enc(rgb)
        gate = self.depth_gating(
            torch.cat([depth, uncertainty], dim=1)
        )
        fused = rgb_feat * (1.0 + gate)
        return self.out_proj(fused)


class DepthDominantExpert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.depth_enc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim // 32, bias=False)
        self.rgb_residual = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim, 1),
        )
        self.unc_blend = nn.Sequential(
            nn.Conv2d(1, dim, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(16, dim),
        )

    def forward(self, rgb, depth, uncertainty):
        depth_feat  = self.depth_enc(depth)
        rgb_res     = self.rgb_residual(rgb)
        unc_blend_w = self.unc_blend(uncertainty)
        fused = depth_feat + unc_blend_w * rgb_res
        return self.out_proj(fused)


class CollaborativeExpert(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.rgb_norm   = nn.LayerNorm(dim)
        self.depth_norm = nn.LayerNorm(dim)

        self.rgb_to_depth = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        self.depth_to_rgb = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

        self.unc_bias_scale = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, depth, uncertainty):
        B, C, h, w = rgb.shape
        N = h * w

        rgb_seq   = rgb.flatten(2).transpose(1, 2)    # [B, N, C]
        depth_seq = depth.flatten(2).transpose(1, 2)  # [B, N, C]

        unc_bias = (uncertainty.flatten(2).transpose(1, 2)
                    * self.unc_bias_scale)             # [B, N, 1]

        rgb_n   = self.rgb_norm(rgb_seq)
        depth_n = self.depth_norm(depth_seq)

        r_attn, _ = self.rgb_to_depth(
            rgb_n,
            depth_n + unc_bias,
            depth_n
        )
        d_attn, _ = self.depth_to_rgb(
            depth_n,
            rgb_n,
            rgb_n
        )

        r_out = rgb_seq   + r_attn
        d_out = depth_seq + d_attn
        fused = self.ffn(torch.cat([r_out, d_out], dim=-1))   # [B, N, C]
        fused = self.out_norm(fused)

        return fused.transpose(1, 2).reshape(B, C, h, w)


class UncertaintyGatedMoE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.expert_r = RGBDominantExpert(dim)
        self.expert_d = DepthDominantExpert(dim)
        self.expert_c = CollaborativeExpert(dim)

        self.router = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self._expert_load = None

    def forward(self, rgb, depth, uncertainty):
        rgb_g   = rgb.mean(dim=[2, 3])          # [B, C]
        depth_g = depth.mean(dim=[2, 3])        # [B, C]
        unc_g   = uncertainty.mean(dim=[2, 3])  # [B, 1]

        router_in  = torch.cat([rgb_g, depth_g, unc_g], dim=1)
        router_out = F.softmax(self.router(router_in), dim=-1)  # [B, 3]

        self._expert_load = router_out.mean(0)  # [3]

        w_r = router_out[:, 0:1].view(-1, 1, 1, 1)
        w_d = router_out[:, 1:2].view(-1, 1, 1, 1)
        w_c = router_out[:, 2:3].view(-1, 1, 1, 1)

        out_r = self.expert_r(rgb, depth, uncertainty)
        out_d = self.expert_d(rgb, depth, uncertainty)
        out_c = self.expert_c(rgb, depth, uncertainty)

        return w_r * out_r + w_d * out_d + w_c * out_c


class EncoderDecoder(nn.Module):
    def __init__(self, num_classes: int = 8, freeze_backbone: bool = True,
                 topk: int = 4):
        super().__init__()
        REPO_DIR = '/media/yuride/date/XZA/toolbox/DINO/dinov3'
        self.backbone = torch.hub.load(
            REPO_DIR, 'dinov3_vitb16',
            source='local',
            weights='/media/yuride/date/XZA/toolbox/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        )
        embed_dim       = 768
        self.num_layers = 12
        self.patch_size = 16
        self.topk       = topk

        for idx in range(self.num_layers):
            self.backbone.blocks[idx] = BlockWithAdapter(
                self.backbone.blocks[idx], embed_dim
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for idx in range(self.num_layers):
                for p in self.backbone.blocks[idx].adapter.parameters():
                    p.requires_grad = True
            print("✓ Backbone frozen, Adapters trainable")

        self.eugm = EUGM(embed_dim)

        self.layer_router = DualStreamIndependentRouter(
            embed_dim, num_layers=self.num_layers, topk=topk
        )

        self.umoe = UncertaintyGatedMoE(embed_dim)

        self.layer_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
                nn.GroupNorm(16, embed_dim),
            )
            for _ in range(topk)
        ])

        self.mlpdecoder = DecoderHead(
            in_channels=[embed_dim] * topk,
            num_classes=num_classes,
            embed_dim=512
        )
        self.upsample16 = nn.Upsample(
            scale_factor=16, mode='bilinear', align_corners=True
        )

    def get_aux_losses(self) -> dict:
        losses = {}

        if hasattr(self, '_nig_params') and self._nig_params is not None:
            losses['nig_reg'] = self._nig_params['nu'].mean()

        r = self.layer_router
        if r._aux_loss_rgb is not None:
            losses['router_balance_rgb']   = r._aux_loss_rgb
        if r._aux_loss_depth is not None:
            losses['router_balance_depth'] = r._aux_loss_depth

        if self.umoe._expert_load is not None:
            load = self.umoe._expert_load
            losses['expert_balance'] = (load * torch.log(load + 1e-6)).sum()

        return losses

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        B, _, H, W = rgb.shape
        h, w    = H // self.patch_size, W // self.patch_size
        N_patch = h * w


        feats_rgb   = self.backbone.get_intermediate_layers(
            rgb, n=self.num_layers, reshape=False
        )
        feats_depth = self.backbone.get_intermediate_layers(
            depth, n=self.num_layers, reshape=False
        )
        total_tokens = feats_rgb[0].shape[1]
        num_special  = total_tokens - N_patch

        cls_rgb   = feats_rgb[-1][:, 0]    # [B, C]
        cls_depth = feats_depth[-1][:, 0]  # [B, C]


        rgb_weights, rgb_topk, depth_weights, depth_topk = \
            self.layer_router(cls_rgb, cls_depth)


        d_last_map = _reshape_token_to_map(
            feats_depth[-1][:, num_special:], h, w
        )
        task_unc_map, nig_params = self.eugm(d_last_map)
        self._nig_params = nig_params


        all_rgb_maps = torch.stack(
            [_reshape_token_to_map(f[:, num_special:], h, w)
             for f in feats_rgb], dim=1
        )
        all_depth_maps = torch.stack(
            [_reshape_token_to_map(f[:, num_special:], h, w)
             for f in feats_depth], dim=1
        )


        arange_B = torch.arange(B, device=rgb.device)
        fused_list = []

        for ki in range(self.topk):
            r_idx = rgb_topk[:, ki]
            d_idx = depth_topk[:, ki]


            rgb_feat_ki   = all_rgb_maps[arange_B, r_idx]    # [B, C, h, w]
            depth_feat_ki = all_depth_maps[arange_B, d_idx]  # [B, C, h, w]

            fused_ki = self.umoe(rgb_feat_ki, depth_feat_ki, task_unc_map)

            lw_r = rgb_weights[arange_B, r_idx].view(B, 1, 1, 1)
            lw_d = depth_weights[arange_B, d_idx].view(B, 1, 1, 1)
            slot_weight = (lw_r * lw_d).sqrt()               # 几何均值
            fused_ki = fused_ki * slot_weight

            fused_ki = self.layer_align[ki](fused_ki)
            fused_list.append(fused_ki)

        out = self.mlpdecoder(fused_list)
        out = self.upsample16(out)
        return out, nig_params


# ─────────────────────────────────────────────────────────────
# 训练示例（辅助损失使用方式）
# ─────────────────────────────────────────────────────────────

def example_train_step(model, rgb, depth, target, criterion, optimizer):
    """
    演示训练循环中的完整损失计算。

    Args:
        rgb, depth : [B, 3, H, W]  网络输入
        target     : [B, H, W]     分割标签
        criterion  : 主损失函数（MscCrossEntropyLoss 等）
        optimizer  : 优化器
    """
    optimizer.zero_grad()

    output, _   = model(rgb, depth)   # nig_params 不需要时可忽略
    seg_loss = criterion(output, target)

    aux         = model.get_aux_losses()
    nig_reg     = aux.get('nig_reg',             torch.tensor(0.0))
    router_rgb  = aux.get('router_balance_rgb',   torch.tensor(0.0))
    router_dep  = aux.get('router_balance_depth', torch.tensor(0.0))
    expert_bal  = aux.get('expert_balance',       torch.tensor(0.0))

    loss = (seg_loss
            + 0.01  * nig_reg
            + 0.001 * (router_rgb + router_dep) / 2.0
            + 0.001 * expert_bal)

    loss.backward()
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("SUIM-D 水下语义分割 v5")
    print("原始Adapter + EUGM(NIG偶然+HAE认知) + DSIR + UMoE")
    print("=" * 60)

    model = EncoderDecoder(num_classes=8, freeze_backbone=True, topk=4)
    model.eval()

    rgb   = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 3, 480, 640)

    with torch.no_grad():
        out, _ = model(rgb, depth)
    print(f"输出 shape : {out.shape}")

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"总参数    : {total:.2f}M")
    print(f"可训练    : {trainable:.2f}M")

    flops,_= profile(model, inputs=(rgb, depth), verbose=False)
    print(f"计算量 (FLOPs) : {flops / 1e9:.2f} GFLOPs")

    # ── 训练模式：验证辅助损失正常（含 nig_reg）──
    model.train()
    with torch.enable_grad():
        out, nig_params = model(rgb, depth)
        aux = model.get_aux_losses()
        print(f"\n辅助损失键: {list(aux.keys())}")
        for k, v in aux.items():
            print(f"  {k}: {v.item():.4f}")

        print(f"\nNIG 参数统计（均值）：")
        for k, v in nig_params.items():
            print(f"  {k:6s}: {v.mean().item():.4f}")

        # 验证双流路由选层
        _, rgb_topk, _, depth_topk = model.layer_router(
            model.backbone.get_intermediate_layers(rgb,  n=12, reshape=False)[-1][:, 0],
            model.backbone.get_intermediate_layers(depth,n=12, reshape=False)[-1][:, 0],
        )
        print(f"\n样本0 RGB  top-k层: {rgb_topk[0].tolist()}")
        print(f"样本0 Depth top-k层: {depth_topk[0].tolist()}")
    print("✅ 运行成功！")