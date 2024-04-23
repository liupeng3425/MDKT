# This is the same as SegFormer but with 256 embed_dims

_base_ = ['dualbb_conv1_mitb5_depth_pretrain_inj.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        backbone_rgb=dict(type='mit_b5_inj_after', used_index = [0,1,2,3]),
    ),
    decode_head=dict(
        type='DAFormerHeadAdvFeature',
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
