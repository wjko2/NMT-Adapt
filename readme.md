
This example uses Spanish as the high resource language and Portuguese as the low resource langauge.
We use English-Spanish parallel corpus to help translating berween English and Portuguese, assuming we have a Portuguese monolingual corpus, but don't have English-Portuguese parallel corpus available.


# English to Low resource language

1. Fine-tune a mBART model for Spanish → English (See https://github.com/pytorch/fairseq/tree/master/examples/mbart about how to use the mBART code)
2. Use fairseq-generate to Feed the Portuguese Monolingual data into the Spanish → English model, to generate backtranslation pairs. 

3. Preprocess data <br>
  * Training data: <br>
    * English → Portuguese (en_XX-pt_XX)<br> Put Backtranslation data in /data/bt/train.en_XX,/data/bt/train.pt_XX  <br>
    * English → Spanish (en_XX-es_XX)<br> Put English-Spanish parallel corpus in /data/para/train.en_XX,/data/para/train.es_XX<br>
    * Spanish denoising (no_ES-es_XX),Portuguese denoising (no_PT-pt_XX)<br> Set the paths and parameters in noise.py and run it<br>
      Put English-Spanish parallel corpus in /data/denoise/train.no_pt1,/data/denoise/train.pt_XX,/data/denoise/train.no_es1,/data/denoise/train.es_XX<br>

  * Testing data: <br>
    * English → Portuguese (en_XX-pt_XX)<br> Put Backtranslation data in /data/test/train.en_XX,/data/test/train.pt_XX  <br>
<br>
 
```


 SPM=XXXXX/sentencepiece/build/src/spm_encode 
 MODEL=XXXXX/mbart.cc25/sentence.bpe.model 
 DATA=/data/bt 
 TRAIN=train 
 VALID=valid 
 SRC=en_XX 
 TGT=pt_XX 
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} 
 #dummy validation
 head -n 1000 ${DATA}/${TRAIN}.spm.${SRC} > ${DATA}/${VALID}.spm.${SRC} 
 head -n 1000 ${DATA}/${TRAIN}.spm.${TGT} > ${DATA}/${VALID}.spm.${TGT}
 DICT=dict.ar_EG.txt 
 DEST=/data 
 NAME=NMTAdapt 
 fairseq-preprocess \ 
   --source-lang ${SRC} \ 
   --target-lang ${TGT} \ 
   --trainpref ${DATA}/${TRAIN}.spm \ 
   --validpref ${DATA}/${VALID}.spm \ 
   --destdir ${DEST}/${NAME} \ 
   --thresholdtgt 0 \ 
   --thresholdsrc 0 \ 
   --srcdict ${DICT} \ 
   --tgtdict ${DICT} \ 
   --workers 70 
 DATA=/data/para 
 SRC=en_XX 
 TGT=es_XX 
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} 
 #dummy validation
 head -n 1000 ${DATA}/${TRAIN}.spm.${SRC} > ${DATA}/${VALID}.spm.${SRC} 
 head -n 1000 ${DATA}/${TRAIN}.spm.${TGT} > ${DATA}/${VALID}.spm.${TGT}
 DICT=dict.ar_EG.txt 
 DEST=/data 
 NAME=NMTAdapt 
 fairseq-preprocess \ 
   --source-lang ${SRC} \ 
   --target-lang ${TGT} \ 
   --trainpref ${DATA}/${TRAIN}.spm \ 
   --validpref ${DATA}/${VALID}.spm \ 
   --destdir ${DEST}/${NAME} \ 
   --thresholdtgt 0 \ 
   --thresholdsrc 0 \ 
   --srcdict ${DICT} \ 
   --tgtdict ${DICT} \ 
   --workers 70 
 DATA=/data/denoise 
 SRC=no_ES1 
 TGT=es_XX 
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} 
 SRC=no_PT1 
 TGT=pt_XX 
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
 ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} 
 python replace.py
 SRC=no_ES 
 TGT=es_XX 
 #dummy validation
 head -n 1000 ${DATA}/${TRAIN}.spm.${SRC} > ${DATA}/${VALID}.spm.${SRC} 
 head -n 1000 ${DATA}/${TRAIN}.spm.${TGT} > ${DATA}/${VALID}.spm.${TGT}
 DICT=dict.ar_EG.txt 
 DEST=/data 
 NAME=NMTAdapt 
 fairseq-preprocess \ 
   --source-lang ${SRC} \ 
   --target-lang ${TGT} \ 
   --trainpref ${DATA}/${TRAIN}.spm \ 
   --validpref ${DATA}/${VALID}.spm \ 
   --destdir ${DEST}/${NAME} \ 
   --thresholdtgt 0 \ 
   --thresholdsrc 0 \ 
   --srcdict ${DICT} \ 
   --tgtdict ${DICT} \ 
   --workers 70 
 SRC=no_PT 
 TGT=pt_XX 
 
 #dummy validation
 head -n 1000 ${DATA}/${TRAIN}.spm.${SRC} > ${DATA}/${VALID}.spm.${SRC} 
 head -n 1000 ${DATA}/${TRAIN}.spm.${TGT} > ${DATA}/${VALID}.spm.${TGT}
 DICT=dict.ar_EG.txt 
 DEST=/data 
 NAME=NMTAdapt 
 fairseq-preprocess \ 
   --source-lang ${SRC} \ 
   --target-lang ${TGT} \ 
   --trainpref ${DATA}/${TRAIN}.spm \ 
   --validpref ${DATA}/${VALID}.spm \ 
   --destdir ${DEST}/${NAME} \ 
   --thresholdtgt 0 \ 
   --thresholdsrc 0 \ 
   --srcdict ${DICT} \ 
   --tgtdict ${DICT} \ 
   --workers 70 
 DATA=/data/test 
 SRC=en_XX 
 TGT=pt_XX 
 TEST=test
 ${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
 ${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} 
DICT=dict.ar_EG.txt 
 DEST=/data 
 NAME=NMTAdapt 
 fairseq-preprocess \ 
   --source-lang ${SRC} \ 
   --target-lang ${TGT} \ 
   --testpref ${DATA}/${TEST}.spm \ 
   --destdir ${DEST}/${NAME} \ 
   --thresholdtgt 0 \ 
   --thresholdsrc 0 \ 
   --srcdict ${DICT} \ 
   --tgtdict ${DICT} \ 
   --workers 70 
```

4. Run our adapt system

Go to directory NMTAdapt1, and load the modified version of fairseq.
(pip install --editable ./)

```
fairseq-train  /data/NMTAdapt   --encoder-normalize-before --decoder-normalize-before   --arch mbart_large --layernorm-embedding   \
--task translation_multi_simple_epoch    --criterion label_smoothed_cross_entropy --label-smoothing 0.2   --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)'\
--lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 50 --total-num-update 80000   --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
--max-tokens 1024 --update-freq 8   --save-interval 1 --save-interval-updates 7000 --keep-interval-updates 10 --no-epoch-checkpoints   --seed 222 --log-format simple\
--log-interval 2  --restore-file /private/home/wjko/fairseq/mbart.cc25/model.pt  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler   \
--ddp-backend no_c10d  --max-epoch 128  --skip-invalid-size-inputs-valid-test --fp16 \
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,pt_XX,no_PT,no_ES\
--lang-pairs en_XX-es_XX,en_XX-pt_XX,no_ES-es_XX,no_PT-pt_XX  --virtual-epoch-size 1388932 --virtual-data-size 138893200 \
--sampling-weights "{'main:en_XX-es_XX': 1,'main:en_XX-pt_XX': 1,'main:no_ES-es_XX': 1,'main:no_PT-pt_XX':1}" --keep-inference-langtok --encoder-langtok src \
--decoder-langtok --lang-tok-style mbart --checkpoint-suffix pt 
```


5.Finetune
```
fairseq-train  /private/home/wjko/data/mcaenes3/mcaenes   --encoder-normalize-before --decoder-normalize-before   --arch mbart_large --layernorm-embedding\
--task translation_multi_simple_epoch    --criterion label_smoothed_cross_entropy --label-smoothing 0.2   --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)'\
--lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 50 --total-num-update 80000   --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0   \
--max-tokens 1024 --update-freq 2   --save-interval 1 --save-interval-updates 3000 --keep-interval-updates 10 --no-epoch-checkpoints   --seed 222 --log-format simple \
--log-interval 2  --restore-file /private/home/wjko/checkpoints/checkpoint_lastpt.pt  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
--ddp-backend no_c10d  --max-epoch 1  --skip-invalid-size-inputs-valid-test --fp16 --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,pt_XX,no_PT,no_ES\
--lang-pairs en_XX-pt_XX  --virtual-epoch-size 846240 --virtual-data-size 138893200 --sampling-weights "{'main:en_XX-pt_XX': 1}" --keep-inference-langtok\
--encoder-langtok src --decoder-langtok --lang-tok-style mbart --checkpoint-suffix pt2
```
6.Testing 
```
fairseq-generate /data/NMTAdapt   --path /checkpoints/checkpoint_lastpt2.pt   --task translation_from_pretrained_bart --gen-subset test\
-t pt_XX -s en_XX   --bpe 'sentencepiece' --sentencepiece-model ${MODEL}   --sacrebleu --remove-bpe 'sentencepiece' --max-tokens 1500\
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,pt_XX,no_PT,no_ES \ > output.txt
```

7.Evaluation <br>
See the evaluation in https://github.com/pytorch/fairseq/tree/master/examples/mbart
# Low resource language to English




Data<br>
    * You have to preprocess Portuguese → English Backtranslation data from the model for English→Portuguese in the latest iteration  <br>
    * Spanish → English <br> Reuse the preprocessed Spanish-English parallel corpus<br>
    * Reuse the preprocessed denoising data <br>
    * Reuse the preprocessed test data <br>
<br>





For this direction, Go to directory NMTAdapt2, and load the modified version of fairseq. (pip install --editable ./)

Training command
```
fairseq-train  /data/NMTAdapt   --encoder-normalize-before --decoder-normalize-before   --arch mbart_large --layernorm-embedding   --task translation_multi_simple_epoch \
--criterion label_smoothed_cross_entropy --label-smoothing 0.2   --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)'   --lr-scheduler polynomial_decay --lr 3e-05 \
--min-lr -1 --warmup-updates 50 --total-num-update 80000   --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0   --max-tokens 1024 --update-freq 8   --save-interval 1 \
--save-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints   --seed 222 --log-format simple --log-interval 2 \
--restore-file /private/home/wjko/fairseq/mbart.cc25/model.pt  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler      --ddp-backend no_c10d \
--max-epoch 55  --skip-invalid-size-inputs-valid-test --fp16 \
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,pt_XX,no_PT,no_ES\
--lang-pairs es_XX-en_XX,pt_XX-en_XX,es_XX-no_ES,pt_XX-no_PT  --virtual-epoch-size 1388932 --virtual-data-size 138893200\
--sampling-weights '{"main:es_XX-en_XX": 1,"main:pt_XX-en_XX": 1,"main:es_XX-no_ES": 1,"main:pt_XX-no_PT":1}' --keep-inference-langtok --encoder-langtok src \
--decoder-langtok --lang-tok-style mbart --checkpoint-suffix pt3
 ```
Testing command
```
fairseq-generate /data/NMTAdapt   --path /checkpoints/checkpoint_lastpt3.pt   --task translation_from_pretrained_bart   --gen-subset test   -t en_XX -s pt_XX \
--bpe 'sentencepiece' --sentencepiece-model ${MODEL}   --sacrebleu --remove-bpe 'sentencepiece'   --max-tokens 1500\
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,pt_XX,no_PT,no_ES\
> output2.txt
```

# Using another High Resource Language
You can use one of those languages included in mBART pretraining as the high resource language
ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

If you want to use a different HRL, you have to make the following modifications to the code:
If you want to use the i th language in the list 

* change the number on line 222,223,224 in /NMTAdapt1/fairseq/models/bart/model.py and /NMTAdapt2/fairseq/models/bart/model.py into i+250000
* change the number on line 80 in /NMTAdapt1/fairseq/criterions/label_smoothed_cross_entropy.py and /NMTAdapt2/fairseq/criterions/label_smoothed_cross_entropy.py into i+250001
* change the number on line 152 in /NMTAdapt1/fairseq/data/language_pair_dataset.py, the number on line 152 and the first number on line 178  in /NMTAdapt2/fairseq/data/language_pair_dataset.py into i+250001


# Citation
```
@InProceedings{ko2021adapting,
  title={Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data},
    author={Wei-Jen Ko and Ahmed El-Kishky and Adithya Renduchintala and Vishrav Chaudhary and Naman Goyal and Francisco Guzmán and Pascale Fung and Philipp Koehn and Mona Diab},
    year={2021},
  booktitle = {ACL},
}
```
