MODELS = ["linear", "mlp", "knn"]

rule all:
    input:
        "results/cas_dataset.csv",
        "results/summary.csv",
        "results/summary.png"

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
    threads:
        min(workflow.cores, 10)
    shell:
        "sklearn_models fit-{wildcards.model_name} {input} -o {output} --n-jobs {threads}"

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
        "results/summary.csv",
        "results/summary.png"
    params:
        labels=" ".join(MODELS)
    shell:
        "evaluate summarize results {input} -l {params.labels}"
