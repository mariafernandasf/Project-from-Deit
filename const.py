ARGS = {
        # saving
        "output_dir" : "ieor6617_output",
        "train_mode" : True,
        "device" : "cpu", # cpu or cuda
        "batch_size": 256, # batch size
        "seed": 0,

        # model training params
        "epochs": 1, # num epochs
        "start_epoch": 0,
        "model" : "deit_small_patch16_224",
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
        "opt": "adamw", # Optimizer, for example adamw TODO: CHANGE TO FUSEDLAMB W CUDA
        "weight_decay": 0.03,
        "momentum": 0.9, # SGD MOMENTUM

        # dataset params
        "data_set": "CIFAR10",
        "input_size" : 224,
        "clip_grad" : None, # clip gradient norm, None = no clipping
        "color_jitter":0.3, # color jitter factor
        "aa": "rand-m9-mstd0.5-inc1", # Use AutoAugment policy. "v0" or "original
        "train_interpolation": "bicubic", # Training interpolation (random, bilinear, bicubic) 
        "reprob": 0.0, # random erase probability
        "remode": "pixel", # random erase mode
        "recount": 1, # random erase count
        "num_workers": 0, # TODO: change to 10
        "ThreeAugment": True, 

        # model eval params
        "eval_crop_ratio": 1.0,
        "eval": False, # SET TO TRUE IF PERFORM EVAL ONLY
        "finetune": "", # checkpoint file IF EVAL
    }