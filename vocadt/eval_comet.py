import os
import json
from dataclasses import dataclass
from comet import download_model, load_from_checkpoint


@dataclass
class EvaluationArguments:
    eval_model: str = "Unbabel/XCOMET-XL"
    eval_batch_size: int = 8
    eval_qe: bool = False
    input_file: str = "outputs/Mistral-7B-VocADT-50k-Latin/flores100.sw-en.5shot.tsv"


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


def main():
    # load tsv file
    eval_args = EvaluationArguments()
    input_file = eval_args.input_file
    output_name = os.path.basename(input_file).replace(".tsv", "")
    output_repo = os.path.dirname(input_file)
    eval_filename = f"{os.path.basename(eval_args.eval_model)}.{output_name}.json"
    eval_filepath = os.path.join(output_repo, eval_filename)
    print(f"Output path: {output_repo}")
    print(f"Output name: {eval_filename}")

    lsrc, lhyp, lref = [], [], []
    with open(input_file, "r") as f:
        for line in f:
            src, hyp, ref = line.strip().split("\t")
            lsrc.append(src)
            lhyp.append(hyp)
            lref.append(ref)
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
