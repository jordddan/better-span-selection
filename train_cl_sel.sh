REPO=$PWD
MODEL=${1:-roberta-base}
TASK=${2:-conll}
GPU=${3:-3}
SEED=${4:-1}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU
SP="--select_positive"
# SP=""
SN="--select_negative"
# SN=""

if [ ${TASK:0:5} == "conll" ]; then
    WARMUP_STEPS=800
    SAVE_STEPS=200
    BATCH_SIZE=24
    BEGIN_EPOCH=1
    BEGIN_F1=0.6
    LR=1e-5
    NUM_EPOCHS=40
elif [ $TASK == "webpage" ]; then
    WARMUP_STEPS=200
    SAVE_STEPS=100
    BATCH_SIZE=16
    BEGIN_EPOCH=1
    BEGIN_F1=0.5
    LR=1e-5
    NUM_EPOCHS=100
elif [ $TASK == "twitter" ]; then
    WARMUP_STEPS=400
    SAVE_STEPS=100
    BATCH_SIZE=32
    BEGIN_EPOCH=1
    BEGIN_F1=0.5
    LR=2e-5
    NUM_EPOCHS=50
elif [ $TASK == "bc5cdr" ]; then
    WARMUP_STEPS=200
    SAVE_STEPS=100
    BATCH_SIZE=32
    BEGIN_EPOCH=1
    BEGIN_F1=0.5
    LR=2e-5
    NUM_EPOCHS=50
fi

if [ $MODEL == "bert-large-cased" ] || [ $MODEL == "bert-base-cased" ]; then
    MODEL_TYPE="bert"
elif [ $MODEL == "roberta-large" ] || [ $MODEL == "roberta-base" ]; then
    MODEL_TYPE="roberta"
fi

GRAD_ACC=1
MLP_DIM=256
MLP_DROPOUT=0.2
SAMPLE_RATE=0.35

DATA_DIR=$DATA_DIR/$TASK
OUTPUT_DIR="$OUT_DIR/${TASK}/${MODEL}-lr${LR}-epoch${NUM_EPOCHS}-warmup${WARMUP_STEPS}-bsz${BATCH_SIZE}-bepoch${BEGIN_EPOCH}-seed${SEED}"
mkdir -p $OUTPUT_DIR
python $REPO/main_cl_sel.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --log_file $OUTPUT_DIR/train.log \
    --labels $DATA_DIR/labels.txt \
    --eval_test_set \
    --do_train --do_eval \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 32 \
    --gradient_accumulation_steps $GRAD_ACC \
    --mlp_hidden_size $MLP_DIM \
    --mlp_dropout_rate $MLP_DROPOUT \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --overwrite_output_dir \
    --seed $SEED \
    --select_begin_epoch $BEGIN_EPOCH \
    --neg_sample_rate $SAMPLE_RATE \
    --pbegin_f1 $BEGIN_F1 \
    $SP $SN \
