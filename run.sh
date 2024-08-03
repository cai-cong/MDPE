#Unimodal
python main.py --feature_set baichuan13B-base --fea_dim 5120
python main.py --feature_set chatglm2-6B --fea_dim 4096
python main.py --feature_set VIT --fea_dim 768
python main.py --feature_set clipVIT-B16 --fea_dim 512
python main.py --feature_set clipVIT-L14 --fea_dim 768
python main.py --feature_set wavlm-base --fea_dim 768
python main.py --feature_set wavlm-large --fea_dim 1024
python main.py --feature_set egemaps --fea_dim 88
python main.py --feature_set chinese-hubert-base --fea_dim 768
python main.py --feature_set chinese-hubert-large --fea_dim 1024
python main.py --feature_set chinese-wav2vec2-base --fea_dim 768
python main.py --feature_set chinese-wav2vec2-large --fea_dim 1024
python main.py --feature_set sbert-chinese-general-v2 --fea_dim 768



#multimodal TA TV AV --fusion
python main.py --feature_set baichuan13B-base --fea_dim 5120 --feature_set2 wavlm-base --fea_dim2 768 --fusion
python main.py --feature_set baichuan13B-base --fea_dim 5120 --feature_set2 VIT --fea_dim2 768 --fusion
python main.py --feature_set wavlm-base --fea_dim 768 --feature_set2 clipVIT-B16 --fea_dim2 512 --fusion

#TAV --AVT --fusion
python main.py --feature_set chatglm2-6B --fea_dim 4096 --feature_set2 VIT --fea_dim2 768 --feature_set3 wavlm-base --fea_dim3 768 --AVT --fusion

#personality --use_personality
python main.py --feature_set baichuan13B-base --fea_dim 5120 --feature_set2 wavlm-base --fea_dim2 768 --use_personality --fusion

#emotion --use_emotion
python main.py --feature_set baichuan13B-base --fea_dim 5120 --use_emotion

#multimodal+personality+emotion
python main.py --feature_set baichuan13B-base --fea_dim 5120 --feature_set2 VIT --fea_dim2 768 --feature_set3 chinese-wav2vec2-base --fea_dim3 768 -fusion --AVT --use_personality --use_emotion -
