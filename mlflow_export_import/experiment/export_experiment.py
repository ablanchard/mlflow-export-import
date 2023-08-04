""" 
Exports an experiment to a directory.
"""

import os
import click
import mlflow

from mlflow_export_import.common.click_options import *
from mlflow_export_import.common import mlflow_utils
from mlflow_export_import.common.iterators import SearchRunsIterator
from mlflow_export_import.common import io_utils
from mlflow_export_import.run.export_run import RunExporter
from mlflow_export_import.common import utils
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class ExperimentExporter():

    def __init__(self, mlflow_client, notebook_formats=None):
        """
        :param mlflow_client: MLflow client.
        :param notebook_formats: List of notebook formats to export. Values are SOURCE, HTML, JUPYTER or DBC.
        """
        self.mlflow_client = mlflow_client
        self.skip_previous_ok_runs = True
        self.run_exporter = RunExporter(self.mlflow_client, notebook_formats=notebook_formats)
        self.save_status_interval = 20

    def _get_previous_ok_runs(self, output_dir):
        manifest_path = os.path.join(output_dir, "experiment.json")
        if not os.path.exists(manifest_path):
            return set()
        manifest = io_utils.read_file(manifest_path)
        if "export_info" in manifest:
            return set(manifest["export_info"]["ok_runs"])
        if "mlflow" in manifest:
            return set(manifest["mlflow"]["runs"])
        raise Exception("Unknown manifest format")
    

    def _save_status(self, output_dir, exp, ok_run_ids, failed_run_ids, futures, total_run):
        print(f"[{datetime.now()} {exp.experiment_id}] Saving status after {total_run} runs")

        # Waiting on futures
        print(f"[{datetime.now()} {exp.experiment_id}] Waiting on {len(futures)} futures")
        for future in futures:
            result = future.result()
            if result[0]:
                ok_run_ids.add(result[1])
            else:
                failed_run_ids.add(result[1])
        
        print(f"[{datetime.now()} {exp.experiment_id}] futures waited")

        info_attr = {
            "num_total_runs": (total_run+1),
            "num_ok_runs": len(ok_run_ids),
            "num_failed_runs": len(failed_run_ids),
            "failed_runs": list(failed_run_ids)
        }
        exp_dct = utils.strip_underscores(exp) 
        exp_dct["tags"] = dict(sorted(exp_dct["tags"].items()))

        mlflow_attr = { "experiment": exp_dct , "runs": list(ok_run_ids) }
        io_utils.write_export_file(output_dir, "experiment.json", __file__, mlflow_attr, info_attr)


    def _log_result_statement(self, exp, ok_run_ids, failed_run_ids, total_run):
        msg = f"for experiment '{exp.name}'"
        prefix = f"[{datetime.now()} {exp.experiment_id}]"
        if len(failed_run_ids) == 0:
            print(f"{prefix} All {len(ok_run_ids)} runs succesfully exported {msg}")
        else:
            print(f"{prefix} {len(ok_run_ids)/total_run} runs succesfully exported {msg}")
            print(f"{prefix} {len(failed_run_ids)/total_run} runs failed {msg}")

    def get_run_id(self, unknow_object):
        if type(unknow_object) == str:
            return unknow_object
        return unknow_object.info.run_id

    def export_experiment(self, exp_id_or_name, output_dir, running_threads, run_ids=None):
        """
        :param exp_id_or_name: Experiment ID or name.
        :param output_dir: Output directory.
        :param run_ids: List of run IDs to export. If None export all run IDs.
        :return: Number of successful and number of failed runs.
        """
        max_workers = int(os.cpu_count() / running_threads)
        print(f"[exp: {exp_id_or_name}] Using {max_workers} threads")
        exp = mlflow_utils.get_experiment(self.mlflow_client, exp_id_or_name)
        print(f"Exporting experiment '{exp.name}' (ID {exp.experiment_id}) to '{output_dir}'")
        failed_run_ids = set()
        j = -1
        ok_run_ids = self._get_previous_ok_runs(output_dir)
        if not run_ids:
            run_ids = SearchRunsIterator(self.mlflow_client, exp.experiment_id)
        
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for j,run in enumerate(run_ids):
                log_prefix = f"[{datetime.now()} {exp.experiment_id}]"
                run_id = self.get_run_id(run)
                if self.skip_previous_ok_runs and run_id in ok_run_ids:
                    print(f"{log_prefix} Skipping run {j+1}: {run_id}")
                else:
                    future = executor.submit(self._export_run, j, run_id, output_dir, log_prefix)
                    futures.append(future)
                # Regularly save status so it's easy to restart
                if len(futures) == self.save_status_interval:
                    self._save_status(output_dir, exp, ok_run_ids, failed_run_ids, futures, j)
                    futures = []

        # Finally save status
        self._save_status(output_dir, exp, ok_run_ids, failed_run_ids, futures, j)

        # Print result
        self._log_result_statement(exp, ok_run_ids, failed_run_ids, j)
        return len(ok_run_ids), len(failed_run_ids) 


    def _export_run(self, idx, run_id, output_dir, log_prefix):
        run_dir = os.path.join(output_dir, run_id)        
        print(f"{log_prefix} Exporting run {idx+1}: {run_id}")
        res = self.run_exporter.export_run(run_id, run_dir)
        return (res, run_id)
        


@click.command()
@opt_experiment
@opt_output_dir
@opt_notebook_formats

def main(experiment, output_dir, notebook_formats):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    client = mlflow.tracking.MlflowClient()
    exporter = ExperimentExporter(
        client,
        notebook_formats=utils.string_to_list(notebook_formats))
    exporter.export_experiment(experiment, output_dir, 1)

if __name__ == "__main__":
    main()
