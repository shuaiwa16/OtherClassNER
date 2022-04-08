rm -f result_dir/*
python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=False \
    --data_dir=data/cluener   \
    --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=3.0   \
    --output_dir=result_dir
