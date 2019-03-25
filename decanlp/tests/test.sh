#!/usr/bin/env bash

set -e
set -x
SRCDIR=`dirname $0`

# functional tests

function delete {
    rm -rf $1
}

mkdir -p $SRCDIR/embeddings
curl -O "https://parmesan.stanford.edu/glove/glove.6B.50d.txt.pt" ; mv glove.6B.50d.txt.pt $SRCDIR/embeddings/
curl -O "https://parmesan.stanford.edu/glove/charNgram.txt.pt" ; mv charNgram.txt.pt $SRCDIR/embeddings/

    TMPDIR=`pwd`
    workdir=`mktemp -d $TMPDIR/decaNLP-tests-XXXXXX`

    i=0

    for hparams_train in "" "--use_curriculum" "--bert_embedding" "--thingpedia $SRCDIR/dataset/thingpedia-8strict.json --almond_grammar full.bottomup" ; do

        hparams_decode=""
        case $hparams_train in
        "--thingpedia"*)
        hparams_decode="--thingpedia $SRCDIR/dataset/thingpedia-8strict.json"
        ;;
        esac

        save_dir=$workdir/model_$i

        # train
        pipenv run decanlp train --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $save_dir --data $SRCDIR/dataset/  --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit $hparams_train

        # greedy decode
        pipenv run decanlp predict --tasks almond --evaluate test --path $save_dir --overwrite --eval_dir $save_dir/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings $hparams_decode

        # export prediction results
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data $SRCDIR/dataset/almond/test.tsv --gold_program $save_dir/eval_results/test/almond.gold.txt --predicted_program $save_dir/eval_results/test/almond.txt --output_file $save_dir/results.tsv

        # check if result files exist
        if [ ! -f $save_dir/results.tsv ] && [ ! -f $save_dir/results_raw.tsv ]; then
            echo "File not found!"
            exit
        fi

        i=$((i+1))
    done

trap "delete $workdir" TERM