python src/train.py \
trainer=gpu datamodule.pin_memory=True datamodule.num_workers=6 datamodule.batch_size=12 \
trainer.max_epochs=20 \
datamodule=celebahq model=gmaa callbacks=val_lastk datamodule.au_mode=random \
model.config_dict.g_part_loss_mode=mse model.config_dict.g_p_part_w=20 \
model.config_dict.g_dt_loss_mode=ssim model.config_dict.g_ssim_dt_w=20 \
model.config_dict.g_bg_loss_mode=ssim model.config_dict.g_ssim_bg_w=20 \
model.config_dict.g_a_w=10 \
logger=tensorboard \
datamodule.test_smallset=False \
"model.config_dict.train_model_name_list=[facenet, ir152, irse50]" \
"model.config_dict.attack_img_root=data/CelebA-pairs/id7256" \
"model.config_dict.attack_name_list=[131714.jpg]" \
"model.config_dict.loss_img_root=data/CelebA-pairs/id7256" \
"model.config_dict.loss_name_list=[000825.jpg]" \
model.config_dict.blackbox=mobile_face \
module_name=gmaa