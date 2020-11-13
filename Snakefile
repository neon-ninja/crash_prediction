MODELS = ["linear", "mlp", "knn"]

rule all:
    input:
        "results/cas_dataset.csv",
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

rule fit_model:
    input:
        "results/cas_dataset.csv"
    output:
        "results/{model_name}_model/model.pickle"
    threads: 10
    shell:
        "sklearn_models fit-{wildcards.model_name} {input} -o {output} -j {threads}"

# check wether we are running in a Slurm job (e.g. on jupyter.nesi.org.nz)
SLURM = "SLURM_NODELIST" in os.environ

rule fit_mlp:
    input:
        "results/cas_dataset.csv"
    output:
        "results/mlp_model/model.pickle"
    threads: 1 if SLURM else workflow.cores
    params:
        n_workers=10 if SLURM else 1,
        threads_per_worker=4 if SLURM else workflow.cores,
        use_slurm="--use-slurm" if SLURM else ""
    shell:
        """
        sklearn_models fit-mlp {input} -o {output} \
            {params.use_slurm} \
            --n-workers {params.n_workers} \
            --threads-per-worker {params.threads_per_worker}
        """

rule predict:
    input:
        "results/cas_dataset.csv",
        "results/{model_name}_model/model.pickle"
    output:
        "results/{model_name}_model/predictions.csv"
    shell:
        "sklearn_models predict {input} -o {output}"

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
