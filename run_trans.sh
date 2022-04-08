#rm -f result_dir_trans/*

python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --do_train=False   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=data/ontonotes   \
    --vocab_file=cased_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=result_dir/model.ckpt-2334   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=3.0   \
    --output_dir=result_dir_trans \
	--other_indice=27 \
	--phase=2 \
	--proto_path=data/ontonotes/proto.pkl \
	--num_others=5
