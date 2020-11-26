USE_SLURM = config.get("USE_SLURM", False)

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
    threads: 1 if USE_SLURM else 8
    params:
        n_workers=lambda wildcards, threads: 50 if USE_SLURM else max(1, threads // 4),
        use_slurm="--use-slurm" if USE_SLURM else ""
    shell:
        """
        models fit {input} {output} --model-type {wildcards.model_type} \
            --n-workers {params.n_workers} {params.use_slurm}
        """

rule fit_knn:
    input:
        "results/cas_dataset.csv"
    output:
        "results/knn_model/model.pickle"
    threads: 1 if USE_SLURM else 8
    params:
        n_workers=lambda wildcards, threads: 10 if USE_SLURM else max(1, threads // 4),
        use_slurm="--use-slurm" if USE_SLURM else ""
    shell:
        """
        models fit {input} {output} --model-type knn \
            --n-workers {params.n_workers} --mem-per-worker "10GB" {params.use_slurm}
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
