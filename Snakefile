USE_SLURM = config.get("USE_SLURM", False)
N_ITER = config.get("N_ITER", 100)

MODELS = ["dummy", "linear", "mlp", "knn", "gbdt"]

rule all:
    input:
        "results/cas_dataset.csv",
        expand("results/{model_name}_model/predictions.csv", model_name=MODELS),
        "results/summary",

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
        cores_per_worker=8 if USE_SLURM else 4,
        n_workers=lambda wildcards, threads: 50 if USE_SLURM else max(1, threads // 4),
        use_slurm="--use-slurm" if USE_SLURM else ""
    shell:
        """
        models fit {input} {output} --model-type {wildcards.model_type} \
             --n-iter {N_ITER} \
             --n-workers {params.n_workers} \
             --cores-per-worker={params.cores_per_worker} \
             {params.use_slurm}
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
             --n-iter {N_ITER} \
             --n-workers {params.n_workers} \
             --mem-per-worker "10GB" \
             {params.use_slurm}
        """

rule fit_gbdt:
    input:
        "results/cas_dataset.csv"
    output:
        "results/gbdt_model/model.pickle"
    threads: 1 if USE_SLURM else 8
    params:
        cores_per_worker=8 if USE_SLURM else 4,
        n_workers=lambda wildcards, threads: 25 if USE_SLURM else max(1, threads // 4),
        use_slurm="--use-slurm" if USE_SLURM else ""
    shell:
        """
        models fit {input} {output} --model-type gbdt \
             --n-iter {N_ITER} \
             --n-workers {params.n_workers} \
             --cores-per-worker={params.cores_per_worker} \
             --mem-per-worker "4GB" \
             {params.use_slurm}
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
        expand("results/{model_name}_model/predictions.csv", model_name=MODELS)
    output:
        directory("results/summary")
    params:
        labels=" ".join(MODELS)
    shell:
        "evaluate {output} {input} -l {params.labels}"