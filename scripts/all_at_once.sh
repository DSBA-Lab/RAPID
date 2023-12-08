
test_size=0.2
unsing_gpu=0

python split_data.py --dataset bgl --test_size $test_size
python split_data.py --dataset tbird --sample 5000000 --test_size $test_size
python split_data.py --dataset hdfs --test_size $test_size

python preprocess_rep.py --dataset bgl --batch_size 8192 --max_token_len 128 --test_size $test_size
python preprocess_rep.py --dataset tbird --sample 5000000 --batch_size 8192 --max_token_len 128 --need_split True --split_num 2 --test_size $test_size
python preprocess_rep.py --dataset hdfs --batch_size 8192 --max_token_len 512 --test_size $test_size

for plm in google/electra-base-discriminator roberta-base
do
    python preprocess_rep.py --dataset bgl --plm $plm --batch_size 8192 --max_token_len 128 --test_size $test_size
    python preprocess_rep.py --dataset tbird --sample 5000000 --plm $plm --batch_size 8192 --max_token_len 128 --need_split True --split_num 2 --test_size $test_size
    python preprocess_rep.py --dataset hdfs --plm $plm --batch_size 8192 --max_token_len 512 --test_size $test_size
done    

for train_ratio in 1
do
    for coreSet in 0.01 0
    do
        for only_cls in False
        do
            echo =====================RQ1, train_ratio:$train_ratio, coreSet:$coreSet, only_cls:$only_cls=====================
            # BGL
            echo BGL    
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset bgl --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #HDFS
            echo HDFS
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset hdfs --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #TBIRD
            echo TBIRD
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset tbird --sample 5000000 --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size
        done
    done
done

for train_ratio in 1 0.1 0.05 0.01 0.001
do
    for coreSet in 0.01 0.1 1 0
    do
        for only_cls in False True
        do
            echo =====================RQ6, train_ratio:$train_ratio, coreSet:$coreSet, only_cls:$only_cls=====================
            # BGL
            echo BGL    
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset bgl --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #HDFS
            echo HDFS
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset hdfs --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #TBIRD
            echo TBIRD
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset tbird --sample 5000000 --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size
        done
    done
done

for train_ratio in 1 0.1 0.05 0.01 0.001
do
    for coreSet in 0 1 2 5 10 0.01 0.05 0.1 0.2 0.3 0.5
    do
        for only_cls in False
        do
            echo =====================RQ2,4, train_ratio:$train_ratio, coreSet:$coreSet, only_cls:$only_cls=====================
            # BGL
            echo BGL    
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset bgl --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #HDFS
            echo HDFS
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset hdfs --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #TBIRD
            echo TBIRD
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset tbird --sample 5000000 --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size
        done
    done
done


for train_ratio in 1 0.9 0.8 0.5 0.3 0.2 0.1 0.05 0.01 0.001
do
    for coreSet in 0 0.01
    do
        for only_cls in False
        do
            echo =====================RQ5, train_ratio:$train_ratio, coreSet:$coreSet, only_cls:$only_cls=====================
            # BGL
            echo BGL    
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset bgl --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #HDFS
            echo HDFS
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset hdfs --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

            #TBIRD
            echo TBIRD
            CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --dataset tbird --sample 5000000 --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size
        done
    done
done

for plm in google/electra-base-discriminator roberta-base
do
    # python preprocess_rep.py --dataset bgl --plm $plm --batch_size 8192 --max_token_len 128 --test_size $test_size
    # python preprocess_rep.py --dataset tbird --sample 5000000 --plm $plm --batch_size 8192 --max_token_len 128 --need_split True --split_num 2 --test_size $test_size
    # python preprocess_rep.py --dataset hdfs --plm $plm --batch_size 8192 --max_token_len 512 --test_size $test_size
    
    for train_ratio in 1
    do
        for coreSet in 0.01
        do
            for only_cls in False
            do
                echo =====================RQ3, plm:$plm, train_ratio:$train_ratio, coreSet:$coreSet, only_cls:$only_cls=====================
                # BGL
                echo BGL    
                CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --plm $plm --dataset bgl --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

                #HDFS
                echo HDFS
                CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --plm $plm --dataset hdfs --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size

                #TBIRD
                echo TBIRD
                CUDA_VISIBLE_DEVICES=$unsing_gpu python ad_test_coreSet.py --plm $plm --dataset tbird --sample 5000000 --train_ratio $train_ratio --coreSet $coreSet --only_cls $only_cls --test_size $test_size
            done
        done
    done
done

