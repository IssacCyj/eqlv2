_base_ = ['./mask_rcnn_r50_8x2_1x.py']

model = dict(roi_head=dict(type='BTRoIHead', bbox_head=dict(type='Shared2FCCBBoxHeadBT', loss_cls=dict(type="EQLv2"), )))

data = dict(train=dict(oversample_thr=1e-3))
# test_cfg = dict(rcnn=dict(max_per_img=800))
# train_cfg = dict(rcnn=dict(sampler=dict(pos_fraction=0.5)))

work_dir = 'bt_1x_rfs'