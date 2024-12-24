import os
import json
import torch
import datasets
import langcodes
import sacrebleu
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from comet import download_model, load_from_checkpoint
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer


two2three_script = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "el": "ell_Grek",
    "ko": "kor_Hang",
    "fi": "fin_Latn",
    "id": "ind_Latn",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "vi": "vie_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "hi": "hin_Deva",
    "et": "est_Latn",  # "ekk_Latn",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "ur": "urd_Arab",
    "sw": "swh_Latn",
    "te": "tel_Telu",
    "eu": "eus_Latn",
    "ht": "hat_Latn",
    "qu": "quy_Latn",
    "sr": "srp_Cyrl",
    "be": "bel_Cyrl",
    "uk": "ukr_Cyrl",
    "mk": "mkd_Cyrl",
    "tg": "tgk_Cyrl",
    "mn": "khk_Cyrl",
    "kk": "kaz_Cyrl",
    "ky": "kir_Cyrl",
    "zh": "cmn_Hans",  # Mandarin Chinese (Standard Beijing, not Taiwanese)
    "ja": "jpn_Jpan",
}

three_script2two = {
    "eng_Latn": "en",
    "rus_Cyrl": "ru",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "ell_Grek": "el",
    "kor_Hang": "ko",
    "fin_Latn": "fi",
    "ind_Latn": "id",
    "tur_Latn": "tr",
    "arb_Arab": "ar",
    "vie_Latn": "vi",
    "bul_Cyrl": "bg",
    "cat_Latn": "ca",
    "hin_Deva": "hi",
    "est_Latn": "et",
    "ben_Beng": "bn",
    "tam_Taml": "ta",
    "urd_Arab": "ur",
    "swh_Latn": "sw",
    "tel_Telu": "te",
    "eus_Latn": "eu",
    "hat_Latn": "ht",
    "quy_Latn": "qu",
    "srp_Cyrl": "sr",
    "bel_Cyrl": "be",
    "ukr_Cyrl": "uk",
    "mkd_Cyrl": "mk",
    "tgk_Cyrl": "tg",
    "khk_Cyrl": "mn",
    "kaz_Cyrl": "kk",
    "kir_Cyrl": "ky",
    "cmn_Hans": "zh",
    "jpn_Jpan": "ja",
}


@dataclass
class Args:
    model_name_or_path: str = "h-j-han/Mistral-7B-VocADT-50k-Latin"
    dataset_path: str = "facebook/flores"
    output_base: str = "outputs"
    src: str = "sw"
    tgt: str = "en"
    nshot: int = 5
    nsample: int = 0  # 0 means all
    text_column_name: str = "sentence"


@dataclass
class GenerationArguments:
    max_new_tokens: int = 256
    max_source_length: int = 1024
    do_sample: bool = True
    num_beams: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    batch_size: int = 1
    split: str = "devtest"
    nshot_split: str = "dev"


@dataclass
class EvaluationArguments:
    eval_model: str = "Unbabel/XCOMET-XL"
    eval_batch_size: int = 8
    eval_qe: bool = False


def comets(
    lsource, lhypothesis, lreference=None, model_name="Unbabel/XCOMET-XL", batch_size=8
):
    if lreference is not None:
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(lsource, lhypothesis, lreference)
        ]
    else:  # qe
        print("No reference provided, Quality Estimation mode.")
        data = [
            {
                "src": src,
                "mt": mt,
            }
            for src, mt in zip(lsource, lhypothesis)
        ]

    # load model
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    model_output = model.predict(data, batch_size=batch_size, gpus=1)
    return model_output


def clean_output_string(output: str, suffix: str, nshot=0):
    try:
        out = output.split(suffix)[1 + nshot].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            print(
                f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}"
            )
            return out[1].strip()
        else:
            print(
                f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}"
            )
            return out[2].strip()
    except:
        print(
            f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix"
        )
    try:
        return output.split(suffix)[2].split("\n")[0].strip()
    except:
        print(
            f"Can not solve the edge case, recover the translation to empty string! The output is:\n{output}"
        )
        return ""


def main():
    (args, gen_args, eval_args) = HfArgumentParser(
        [Args, GenerationArguments, EvaluationArguments]
    ).parse_args_into_dataclasses()
    src_lang = args.src
    tgt_lang = args.tgt
    src_lang_full_name = langcodes.Language.get(src_lang).display_name("en")
    tgt_lang_full_name = langcodes.Language.get(tgt_lang).display_name("en")
    lpair = f"{src_lang}-{tgt_lang}"

    # if the number of "/" is more than one, then output repo is the same as the model path
    if args.model_name_or_path.count("/") > 1:
        output_repo = args.model_name_or_path
    else:
        output_repo = os.path.join(
            args.output_base,
            os.path.basename(args.model_name_or_path),
        )
    os.makedirs(output_repo, exist_ok=True)
    output_name = f'{os.path.basename(args.dataset_path)}{"" if args.nsample == 0 else args.nsample}.{lpair}.{args.nshot}shot'
    print(f"Output path: {output_repo}")
    print(f"Output name: {output_name}")
    output_path = os.path.join(output_repo, f"{output_name}.tsv")
    eval_filename = f"{os.path.basename(eval_args.eval_model)}.{output_name}"
    eval_filepath = os.path.join(output_repo, f"{eval_filename}.json")

    # Load base model and LoRA weights
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="left"
    )
    tokenizer.pad_token_id = (
        0 if "<pad>" not in tokenizer.get_vocab() else tokenizer.pad_token_id
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.pad_token_id = (
        0 if "<pad>" not in tokenizer.get_vocab() else tokenizer.pad_token_id
    )
    model.generation_config.pad_token_id = (
        0 if "<pad>" not in tokenizer.get_vocab() else tokenizer.pad_token_id
    )

    dataset_src = datasets.load_dataset(
        args.dataset_path,
        f"{two2three_script[src_lang]}",
        split=(
            gen_args.split
            if args.nsample == 0
            else f"{gen_args.split}[:{args.nsample}]"
        ),
    )
    dataset_tgt = datasets.load_dataset(
        args.dataset_path,
        f"{two2three_script[tgt_lang]}",
        split=(
            gen_args.split
            if args.nsample == 0
            else f"{gen_args.split}[:{args.nsample}]"
        ),
    )
    if args.nshot > 0:
        dataset_src_nshot = datasets.load_dataset(
            args.dataset_path,
            f"{two2three_script[src_lang]}",
            split=(gen_args.nshot_split),
        )
        dataset_tgt_nshot = datasets.load_dataset(
            args.dataset_path,
            f"{two2three_script[tgt_lang]}",
            split=(gen_args.nshot_split),
        )
        lsrc_nshot = []
        lref_nshot = []
        for i, (src_item_nshot, tgt_item_nshot) in enumerate(
            zip(dataset_src_nshot, dataset_tgt_nshot)
        ):
            assert src_item_nshot["id"] == tgt_item_nshot["id"]
            src_nshot = src_item_nshot[args.text_column_name]
            tgt_nshot = tgt_item_nshot[args.text_column_name]
            lsrc_nshot.append(src_nshot)
            lref_nshot.append(tgt_nshot)

    lsrc = []
    lref = []
    lhyp = []
    for i, (src_item, tgt_item) in enumerate(zip(tqdm(dataset_src), dataset_tgt)):
        src = src_item[args.text_column_name]
        tgt = tgt_item[args.text_column_name]
        assert src_item["id"] == tgt_item["id"]
        lsrc.append(src)
        lref.append(tgt)

        prefix = ""
        if args.nshot == 0:
            main = f"{src_lang_full_name}: {src}"
        else:
            # among the n-shot examples len(lsrc_nshot), choose args.nshot index randomly
            nshot_index = np.random.choice(len(lsrc_nshot), args.nshot)
            main = ""
            for j in nshot_index:
                main += f"{src_lang_full_name}: {lsrc_nshot[j]}\n{tgt_lang_full_name}: {lref_nshot[j]}\n"
            main += f"{src_lang_full_name}: {src}"
        suffix = f"\n{tgt_lang_full_name}:"

        prompt = f"{prefix}{main}{suffix}"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # True
            max_length=gen_args.max_source_length,
            truncation=True,
        )
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids.cuda(),
                attention_mask=inputs.attention_mask.cuda(),
                num_beams=gen_args.num_beams,
                max_new_tokens=gen_args.max_new_tokens,
                do_sample=gen_args.do_sample,
                temperature=gen_args.temperature,
                top_p=gen_args.top_p,
            )
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        hyp = clean_output_string(outputs[0], suffix, nshot=args.nshot)
        try:
            output = outputs[0].split(prompt)[1]
        except:
            print("output does not contain prompt as prefix")
            print("Save output as a whole")
            output = outputs[0]
        lhyp.append(hyp)
        if i % 10 == 0:
            print(
                f"i: {i}\{len(dataset_src)}\n[OUTPUT]:\n{output}\n[SRC]: {src}\n[REF]: {tgt}\n[HYP]: {hyp}"
            )
    columns = ["src", "ref", "hyp"]
    df = pd.DataFrame(list(zip(lsrc, lref, lhyp)), columns=columns)
    df.to_csv(output_path, sep="\t", index=False)
    bleu = sacrebleu.corpus_bleu(lhyp, [lref])
    bleu_score = bleu.score
    print(f"BLEU: {bleu_score}")

    # Please check in advance if eval_models are available and authorized to use.
    eval_outputs = comets(
        lsrc,
        lhyp,
        lref if not eval_args.eval_qe else None,
        model_name=eval_args.eval_model,
        batch_size=eval_args.eval_batch_size,
    )
    eval_score = eval_outputs.system_score
    print(f"{eval_args.eval_model}: {eval_score}")

    with open(eval_filepath, "w") as f:
        json.dump(eval_outputs, f)


if __name__ == "__main__":
    main()
