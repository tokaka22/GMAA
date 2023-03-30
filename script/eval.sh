python src/eval.py \
datamodule=celebahq model=gmaa_eval datamodule.test_smallset=False datamodule.data_split=train_test \
model.config_dict.conti_au_path=data/typical_au.txt \
ckpt_path="trained model ckpt path here" \
module_name=gmaa
