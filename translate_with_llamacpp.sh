#!/bin/bash

seed=42
threads=8
temp=0.001
top_k=20
top_p=0.8
min_p=0.0
presence_penalty=0.0
############################################

for presence_penalty in 0.0
do
    for lp in en-fr
    do
        src=${lp%%-*}
        tgt=${lp#*-}
        for model_full in \
            Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
            unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf \
            Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf \
            Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
            Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf
        do
            author_hf=$(echo "$model_full" | cut -d'/' -f1)
            model=$(echo "$model_full" | cut -d'/' -f2)
            model_version=$(basename $model_full)
            model_version_stripped=${model_version%.gguf}
            curl --output $model_version -L https://huggingface.co/$author_hf/$model/resolve/main/$model_version?download=true
            for test in wmt24pp-head50
            do

                python ../biomed_data_set/translate_with_llamacpp.py --input ../biomed_data_set/wmt24pp-head50.$src.txt --src-lang $src --tgt-lang $tgt \
                --output ../biomed_data_set/wmt24pp-head50.$model_version_stripped.$tgt.txt --model $model_full --llama-cpp-path ./build/bin/llama-cli \
                --temp $temp --top-k $top_k --top-p $top_p --min-p $min_p --presence-penalty $presence_penalty &> ../biomed_data_set/wmt24pp-head50.$model_version_stripped.$tgt.log

            done
        done
    done
done

pushd ../biomed_data_set
tar -czvf wmt24pp-head50.tar.gz wmt24pp-head50.*
popd