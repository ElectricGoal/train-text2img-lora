accelerate launch train_img2img_lora.py --config config.yml mixed_precision=bf16

# # With CLI overrides (OmegaConf dotlist syntax)
# python train_img2img_lora.py --config config.yml resolution=768 learning_rate=1e-5 train_batch_size=4

# # With accelerate
# accelerate launch train_img2img_lora.py --config config.yml mixed_precision=bf16