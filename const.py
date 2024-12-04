ARGS = {
        # distributed
        "distributed": False, 
        "dist_url": 'env://',
        "repeated_aug": True,
        "dist_eval": True, 

        # model params that need to be changed for eval
        "eval_crop_ratio": 1.0,
        "eval": False, # SET TO TRUE IF PERFORM EVAL ONLY
        "finetune": "", # checkpoint file IF EVAL i.e. ieor6617_output/rope_axial_deit_small_patch16_LS/checkpoint.pth
        "batch_size": 256, # batch size for training: 256, batch size for eval: 128
        "input_size" : 224, # 224 for training

        "output_dir" : "ieor6617_output/",
        "train_mode" : True,
        "device" : "cuda", # cpu or cuda
        "seed": 0,
        "pin_mem": False, # Pin CPU memory in DataLoader for more efficient transfer to GPU
        
        # model training params
        "resume": "", # resume from checkpoint i.e. ieor6617_output/rope_axial_ape_deit_small_patch16_LS/checkpoint.pth
        "start_epoch": 0, # epoch to resume from
        "epochs": 100, # num epochs
        "model" : "performer_rope_axial_deit_small_patch16_LS",
        "drop"  : 0.0, # dropout rate
        "drop_path" : 0.0, # dropout path rate
        "bce_loss": True,
        "smoothing": 0.0, # label smoothing
        "mixup": 0.8, # mixup aplha, mixup enabled if > 0.
        "cutmix": 1.0, # cutmix alpha, cutmix enabled if > 0.
        "cutmix_minmax": None, # cutmix min/max ratio, overrides alpha and enables cutmix if set
        "mixup_prob": 1.0, # Probability of performing mixup or cutmix when either/both is enabled
        "mixup_switch_prob": 0.5, # Probability of switching to cutmix when both mixup and cutmix enabled
        "mixup_mode": "batch", # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
        "unscale_lr": True, 
        "lr": 4e-3, # learning rate
        "opt": "fusedlamb", # Optimizer, for example adamw or fusedlamb
        "weight_decay": 0.03,
        "momentum": 0.9, # SGD MOMENTUM
        "sched": "cosine", # LR scheduler, default is cosine

        # dataset params
        "data_set": "CIFAR10",
        "clip_grad" : None, # clip gradient norm, None = no clipping
        "color_jitter":0.3, # color jitter factor
        "aa": "rand-m9-mstd0.5-inc1", # Use AutoAugment policy. "v0" or "original
        "train_interpolation": "bicubic", # Training interpolation (random, bilinear, bicubic) 
        "reprob": 0.0, # random erase probability
        "remode": "pixel", # random erase mode
        "recount": 1, # random erase count
        "num_workers": 10, 
        "ThreeAugment": True, 
    }