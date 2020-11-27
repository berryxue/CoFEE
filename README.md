# Coarse-to-Fine Pre-training for Named Entity Recognition
====

Models and results can be found at our EMNLP 2020 paper [Coarse-to-Fine Pre-training for Named Entity Recognition](https://www.aclweb.org/anthology/2020.emnlp-main.514.pdf). It achieves the state-of-the-art result on three benchmarks of NER task.

Details will be updated soon.

Requirement:
======
	Python: 3.8.5
	PyTorch: 1.6.0

Data preporation:
======

***Data for ESI***

You can donwload the data for ESI used in our paper from [HERE](https://drive.google.com/drive/folders/1qE-4P0SH8qHamPHwmlX2z_0cZQnc1bXd?usp=sharing)

If you wanna to generate your own data, please run "data_preprocess/write_tag.py"

***Data for NEE***

You can donwload the data for NEE used in our paper from [HERE](https://drive.google.com/drive/folders/1-2dPlo1iLhKWHzdicS01RxEaipSQwgia?usp=sharing)

Gazetteers used in our paper can be downloaded from [HERE](https://drive.google.com/drive/folders/1COlXFWFIWrN8nHeE7HUHdso5m5IK3T33?usp=sharing)

If you wanna to generate your own data, please run "data_preprocess/write_tag_from_dict.py"

***Query Generation***

Write down queries for entity labels in `./data_preprocess/dump_query2file.py` and run `python3 ./data_preprocess/dump_query2file.py` to dump queries to the folder `./data_preprocess/queries`. 

***Transform tagger-style annotations to MRC-style triples*** 

Run `./data_preprocess/example/generate_data.py` to generate MRC-style data. 

How to run the code?
====

***Stage1: ESI***

python -u run/train_bert_mrc.py --data_dir data/train_for_ESI/zhwiki/ --checkpoint 20000 --learning_rate 6e-6 --num_train_epochs 5 --output_dir data/saved_model/zhwiki/morethan3/5-6e-6 --data_sign zhwiki

***Stage2: NEE***

python -u run/train_bert_mrc.py --data_dir data/train_for_ESI/ecommerce/ --checkpoint 4000 --learning_rate 3e-5 --num_train_epochs 6 --output_dir data/saved_model/ecommerce/3e-5-6-regenerate0.1-perepoch0-warmup0.4 --data_sign ecommerce --pretrain data/saved_model/zhwiki/morethan3/5-6e-6 --bert_model data/bert_model/bert-base-chinese-pytorch/ --warmup_proportion 0.4 --regenerate_rate 0.1 --STrain 1 --perepoch 0

***Stage3: FET***

python -u run/train_cluster_bert_mrc.py --data_dir data/train_for_ESI/ecommerce/pretrain/ --checkpoint 2000 --learning_rate 2e-5 --num_train_epochs 5 --output_dir data/saved_model/ecommerce/cluster/2e-5-5-23-0.001-sumpool-drop0.1-60 --data_sign ecommerce --pretrain data/saved_model/ecommerce/3e-5-6-regenerate0.1-perepoch0-warmup0.4 --bert_model data/bert_model/bert-base-chinese-pytorch/ --num_clusters 23 --gama 0.001 --clus_niter 60 --dropout_rate 0.1 

***Stage4: fine-tune***

python -u run/train_bert_mrc.py --data_dir data/supervised_data/ecommerce/ --checkpoint 100 --learning_rate 3e-5 --num_train_epochs 25 --output_dir data/saved_model/ecommerce/supervised/3e-5-25 --data_sign ecommerce --pretrain data/saved_model/ecommerce/cluster/2e-5-5-23-0.001-sumpool-drop0.1-6 --bert_model data/bert_model/bert-base-chinese-pytorch/

Cite: 
========

@inproceedings{mengge-etal-2020-coarse,

    title = "{C}oarse-to-{F}ine {P}re-training for {N}amed {E}ntity {R}ecognition",
    
    author = "Mengge, Xue  and
    
      Yu, Bowen  and
      
      Zhang, Zhenyu  and
      
      Liu, Tingwen  and
      
      Zhang, Yue  and
      
      Wang, Bin",
      
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    
    year = "2020",
    
    pages = "6345--6354",
    
    }
