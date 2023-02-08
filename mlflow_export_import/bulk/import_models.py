"""
Imports models and their experiments and runs.
"""

import os
import time
import json
import click
from concurrent.futures import ThreadPoolExecutor

import mlflow
from mlflow_export_import.common.click_options import *
from mlflow_export_import.common import io_utils
from mlflow_export_import.experiment.import_experiment import ExperimentImporter
from mlflow_export_import.model.import_model import AllModelImporter


EXPERIMENT_FOLDER = "experiments-2023-01-17"

def _remap(run_info_map):
    res = {}
    for dct in run_info_map.values():
        for src_run_id,run_info in dct.items():
            res[src_run_id] = run_info
    return res


def _import_experiments(client, input_dir, use_src_user_id):
    start_time = time.time()

    dct = io_utils.read_file_mlflow(os.path.join(os.path.join(input_dir,EXPERIMENT_FOLDER,"experiments.json")))
    exps = dct["experiments"]

    importer = ExperimentImporter(client, use_src_user_id=use_src_user_id)
    print("Experiments:")
    for exp in exps: 
        print(" ",exp)
    run_info_map = {}
    exceptions = []

    for exp in exps: 
        exp_input_dir = os.path.join(input_dir, EXPERIMENT_FOLDER, exp["id"])
        try:
            _run_info_map = importer.import_experiment( exp["name"], exp_input_dir)
            run_info_map[exp["id"]] = _run_info_map
        except Exception as e:
            exceptions.append(str(e))
            import traceback
            traceback.print_exc()

    duration = round(time.time()-start_time, 1)
    if len(exceptions) > 0:
        print(f"Errors: {len(exceptions)}")
    print(f"Duration: {duration} seconds")

    return run_info_map, { "experiments": len(exps), "exceptions": exceptions, "duration": duration }


def _import_models(client, input_dir, run_info_map, delete_model, import_source_tags, verbose, use_threads):
    max_workers = int(os.environ.get("CPU_COUNT", os.cpu_count() or 4)) if use_threads else 1
    start_time = time.time()

    models_dir = os.path.join(input_dir, "models")
    models = io_utils.read_file_mlflow(os.path.join(models_dir,"models.json"))
    models = models["models"]
    if os.path.exists(os.path.join(input_dir,"import-conf.json")):
        conf = io_utils.read_file(os.path.join(input_dir,"import-conf.json"))
    else:
        conf = {}
    importer = AllModelImporter(client, run_info_map=run_info_map, import_source_tags=import_source_tags, conf=conf)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model in models:
            dir = os.path.join(models_dir, model)
            executor.submit(importer.import_model, model, dir, delete_model, verbose)

    duration = round(time.time()-start_time, 1)
    return { "models": len(models), "duration": duration }

def read_imported_experiments(input_dir):
    dct = io_utils.read_file_mlflow(os.path.join(os.path.join(input_dir,EXPERIMENT_FOLDER,"experiments.json")))
    exps = dct["experiments"]

    results = {}
    for exp in exps:
        try:
            imported = io_utils.read_file(os.path.join(os.path.join(input_dir,EXPERIMENT_FOLDER, exp["id"],"import-experiment.json")))
        except FileNotFoundError:
            print(f"ERROR no import-experiment.json for experiment {exp['id']}")
            continue
        results = { **results, **imported["runs"] }
    return results


def import_all(client, input_dir, delete_model, use_src_user_id=False, import_source_tags=False, verbose=False, use_threads=False):
    start_time = time.time()
    #exp_res = _import_experiments(client, input_dir, use_src_user_id)
    #run_info_map = _remap(exp_res[0])
    run_info_map = read_imported_experiments(input_dir)
    model_res = _import_models(client, input_dir, run_info_map, delete_model, import_source_tags, verbose, use_threads)
    duration = round(time.time()-start_time, 1)
    dct = { "duration": duration, "model_import": model_res }
    io_utils.write_file("import_report.json", dct)
    print("\nImport report:")
    print(json.dumps(dct,indent=2)+"\n")


@click.command()
@opt_input_dir
@opt_delete_model
@opt_use_src_user_id
@opt_verbose
@opt_import_source_tags
@opt_use_threads

def main(input_dir, delete_model, use_src_user_id, import_source_tags, verbose, use_threads):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    client = mlflow.tracking.MlflowClient()
    import_all(
        client,
        input_dir, 
        delete_model=delete_model, 
        use_src_user_id=use_src_user_id, 
        import_source_tags=import_source_tags,
        verbose=verbose, 
        use_threads=use_threads)


if __name__ == "__main__":
    main()
