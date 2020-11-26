SLURM = config.get("SLURM", False)

MODELS = ["linear", "mlp", "knn", "rf"]

rule all:
    input:
        "results/cas_dataset.csv",
        expand("results/{model_name}_model/scores.csv", model_name=MODELS),
        multiext("results/summary", ".csv", ".png")

rule download_data:
    output:
        "data/cas_dataset.csv"
    shell:
        "cas_data download {output}"

rule prepare_data:
    input:
        "data/cas_dataset.csv"
    output:
        "results/cas_dataset.csv"
    shell:
        "cas_data prepare {input} -o {output}"

rule fit:
    input:
        "results/cas_dataset.csv"
    output:
        "results/{model_type}_model/model.pickle"
    threads: 1 if SLURM else 10
    params:
        n_iter=50 if SLURM else 10,
        slurm_config="-s config/mlp.yaml" if SLURM else ""
    shell:
        """
        models fit-mlp {input} {output} -j {threads} -m {wildcards.model_type} \
            -n {params.n_iter} {params.slurm_config}
        """

rule predict:
    input:
        "results/cas_dataset.csv",
        "results/{model_name}_model/model.pickle"
    output:
        "results/{model_name}_model/predictions.csv"
    shell:
        "models predict {input} -o {output}"

rule evaluate:
    input:
        "results/cas_dataset.csv",
        "results/{model_name}_model/predictions.csv"
    output:
        "results/{model_name}_model/scores.csv",
        "results/{model_name}_model/curves.png"
    shell:
        "evaluate score {input} results/{wildcards.model_name}_model"

rule summarize:
    input:
        expand("results/{model_name}_model/scores.csv", model_name=MODELS)
    output:
        multiext("results/summary", ".csv", ".png")
    params:
        labels=" ".join(MODELS)
    shell:
        "evaluate summarize results {input} -l {params.labels}"
