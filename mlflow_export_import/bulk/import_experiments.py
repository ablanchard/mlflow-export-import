""" 
Import a list of experiment from a directory.
"""

import os
import click
from concurrent.futures import ThreadPoolExecutor

import mlflow

from mlflow_export_import.common.click_options import *
from mlflow_export_import.common import io_utils
from mlflow_export_import.experiment.import_experiment import ExperimentImporter


def _import_experiment(importer, exp_name, exp_input_dir):
    try:
        importer.import_experiment(exp_name, exp_input_dir)
    except Exception:
        import traceback
        traceback.print_exc()


def import_experiments(client, input_dir, use_src_user_id=False, nb_threads_all=1, threads=12, save_interval=20000, experiments="all", exclude_experiments="", filter_user=""): 
    dct = io_utils.read_file_mlflow(os.path.join(input_dir, "experiments.json"))
    experiments_to_do = dct["experiments"]

    if experiments != "all":
        print(f"Will only do: {experiments}")
        experiment_ids_to_do = experiments.split(",")
        experiments_to_do = list(filter(lambda exp: exp["id"] in experiment_ids_to_do, experiments_to_do))

    if exclude_experiments != "":
        print(f"Will exclude: {exclude_experiments}")
        experiments_ids_to_exclude = exclude_experiments.split(",")
        experiments_to_do = list(filter(lambda exp: exp["id"] not in experiments_ids_to_exclude, experiments_to_do))

    if filter_user != "":
        print(f"Will only run on {filter_user} experiments")
        experiments_to_do = list(filter(lambda exp: filter_user in exp["name"], experiments_to_do))
    
    for exp in experiments_to_do:
        print("  ", exp)


    if os.path.exists(os.path.join(input_dir,"import-conf.json")):
        conf = io_utils.read_file(os.path.join(input_dir, "..","import-conf.json"))
    else:
        conf = {}

    max_workers = nb_threads_all
    threads_per_experiment = threads / nb_threads_all
    importer = ExperimentImporter(client, use_src_user_id=use_src_user_id, conf=conf, save_status_interval=save_interval, threads=threads_per_experiment)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for exp in experiments_to_do:
            exp_input_dir = os.path.join(input_dir,exp["id"])
            exp_name = exp["name"]
            executor.submit(_import_experiment, importer, exp_name, exp_input_dir)


@click.command()
@opt_input_dir
@opt_use_src_user_id
@opt_nb_threads_all
@opt_threads
@opt_save_interval
@opt_experiments
@opt_exclude_experiments
@opt_filter_user
def main(input_dir, use_src_user_id, nb_threads_all, threads, save_interval, experiments, exclude_experiments, filter_user): 
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    client = mlflow.tracking.MlflowClient()
    import_experiments(client, input_dir, use_src_user_id,  nb_threads_all, threads, save_interval, experiments, exclude_experiments, filter_user)

if __name__ == "__main__":
    main()
