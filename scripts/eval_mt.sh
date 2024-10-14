OUTPUT_DIR=outputs
mkdir -p $OUTPUT_DIR


# Latin
MODEL=h-j-han/Mistral-7B-VocADT-50k-Latin
TGT=en
for SRC in sw id et ht ; do
    python vocadt/decode_llm_mt.py --model_name_or_path=$MODEL --src=$SRC --tgt=$TGT 
done

SRC=en
for TGT in sw id et ht ; do
    python vocadt/decode_llm_mt.py --model_name_or_path=$MODEL --src=$SRC --tgt=$TGT 
done


# Mixed
MODEL=h-j-han/Mistral-7B-VocADT-50k-Mixed
TGT=en
for SRC in ko el ru bg ; do
    python vocadt/decode_llm_mt.py --model_name_or_path=$MODEL --src=$SRC --tgt=$TGT 
done

SRC=en
for TGT in ko el ru bg ; do
    python vocadt/decode_llm_mt.py --model_name_or_path=$MODEL --src=$SRC --tgt=$TGT 
done


# Cyrillic
MODEL=h-j-han/Mistral-7B-VocADT-50k-Cyrillic
TGT=en
for SRC in uk kk ; do
    python vocadt/decode_llm_mt.py --model_name_or_path=$MODEL --src=$SRC --tgt=$TGT 
done

SRC=en
for TGT in uk kk ; do
    python vocadt/decode_llm_mt.py --model_name_or_path=$MODEL --src=$SRC --tgt=$TGT 
done