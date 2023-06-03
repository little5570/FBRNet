cfg = dict(
    model_type='mdoel',
    n_cats=19,
    num_aux_heads=4,
#     lr_start=1e-2,
#     lr_start=1e-3,
    lr_start=1e-3,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    dataset='CityScapes',
    im_root='/powerhope/wangzhuo/cityscapes',
    train_im_anns='/powerhope/wangzhuo/cityscapes/train.txt',
    val_im_anns='/powerhope/wangzhuo/cityscapes/val.txt',
    test_im_anns='/powerhope/wangzhuo/cityscapes/test.txt',
 
  
    scales=[0.75, 2.],
    cropsize=[1024, 1024],
    eval_crop=[1024, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=4,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
