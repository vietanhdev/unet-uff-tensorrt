
configs = []

# Batch size: 8
# Image size: 448

configs.append({
    "name": "config03",
    "LR": 0.001,
    "loss": "focal+dice",
    "freeze_back": False,
    "filters": (128, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config03/model_.127-0.371112.h5"
})

configs.append({
    "name": "config04",
    "LR": 0.001,
    "loss": "focal+tversky",
    "freeze_back": False,
    "filters": (128, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config04/model_.113-0.464030.h5"
})

configs.append({
    "name": "config05",
    "LR": 0.001,
    "loss": "binary_ce",
    "freeze_back": False,
    "filters": (128, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config05/model_.116-0.024944.h5"
})

configs.append({
    "name": "config13",
    "LR": 0.001,
    "loss": "focal+dice",
    "freeze_back": False,
    "filters": (128, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config13/model_.112-0.386539.h5"
})

configs.append({
    "name": "config14",
    "LR": 0.001,
    "loss": "focal+tversky",
    "freeze_back": False,
    "filters": (128, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config14/model_.110-0.482715.h5"
})

configs.append({
    "name": "config15",
    "LR": 0.001,
    "loss": "binary_ce",
    "freeze_back": False,
    "filters": (128, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config15/model_.135-0.025376.h5"
})



configs.append({
    "name": "config16",
    "LR": 0.001,
    "loss": "focal+dice",
    "freeze_back": False,
    "filters": (64, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config16/model_.141-0.383609.h5"
})

configs.append({
    "name": "config17",
    "LR": 0.001,
    "loss": "focal+tversky",
    "freeze_back": False,
    "filters": (64, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config17/model_.073-0.483661.h5"
})

configs.append({
    "name": "config18",
    "LR": 0.001,
    "loss": "binary_ce",
    "freeze_back": False,
    "filters": (64, 64, 32, 16, 8),
    "model_path": "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config18/model_.077-0.026460.h5"
})

