BATCH_SIZE=8
DTYPE=bfloat16
OUTPUT_DIR=outputs
mkdir -p $OUTPUT_DIR

# Install lm eval from https://github.com/EleutherAI/lm-evaluation-harness

# Latin
MODEL=h-j-han/Mistral-7B-VocADT-50k-Latin

TASKS=xnli_sw,xcopa_sw,xcopa_id,xcopa_et,xcopa_ht
NSHOT=0
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=$DTYPE \
    --tasks $TASKS \
    --num_fewshot $NSHOT \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \

TASKS=belebele_swh_Latn,belebele_ind_Latn,belebele_est_Latn,belebele_hat_Latn,m_mmlu_id
NSHOT=5
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=$DTYPE \
    --tasks $TASKS \
    --num_fewshot $NSHOT \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \


# Mixed
MODEL=h-j-han/Mistral-7B-VocADT-50k-Mixed

TASKS=xnli_el,xnli_ru,xnli_bg,xnli_en
NSHOT=0
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=$DTYPE \
    --tasks $TASKS \
    --num_fewshot $NSHOT \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \


TASKS=belebele_kor_Hang,belebele_ell_Grek,belebele_rus_Cyrl,belebele_bul_Cyrl,belebele_eng_Latn,m_mmlu_ru,m_mmlu_en
NSHOT=5
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=$DTYPE \
    --tasks $TASKS \
    --num_fewshot $NSHOT \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \


# Cyrillic
MODEL=h-j-han/Mistral-7B-VocADT-50k-Cyrillic


TASKS=belebele_ukr_Cyrl,belebele_kaz_Cyrl,,m_mmlu_uk
NSHOT=5
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=$DTYPE \
    --tasks $TASKS \
    --num_fewshot $NSHOT \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \