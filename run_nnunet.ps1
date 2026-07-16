$env:nnUNet_raw          = "d:/Medical-Image-Segmentation/finetune/fileStorage/nnunet_env/nnUNet_raw"
$env:nnUNet_preprocessed = "d:/Medical-Image-Segmentation/finetune/fileStorage/nnunet_env/nnUNet_preprocessed"
$env:nnUNet_results      = "d:/Medical-Image-Segmentation/finetune/fileStorage/nnunet_env/nnUNet_results"
$env:nnUNet_n_proc_DA    = "4"
& "d:/Medical-Image-Segmentation/.venv-train/Scripts/nnUNetv2_train.exe" 1 3d_fullres 0 --npz
