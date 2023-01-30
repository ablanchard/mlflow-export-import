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


class ExperimentExporter():

    def __init__(self, mlflow_client, notebook_formats=None):
        """
        :param mlflow_client: MLflow client.
        :param notebook_formats: List of notebook formats to export. Values are SOURCE, HTML, JUPYTER or DBC.
        """
        self.mlflow_client = mlflow_client
        self.skip_previous_ok_runs = True
        self.run_exporter = RunExporter(self.mlflow_client, notebook_formats=notebook_formats)

    def _get_previous_ok_runs(self, output_dir):
        manifest_path = os.path.join(output_dir, "experiment.json")
        if not os.path.exists(manifest_path):
            return []
        manifest = io_utils.read_file(manifest_path)
        if "export_info" in manifest:
            return manifest["export_info"]["ok_runs"]
        if "mlflow" in manifest:
            return manifest["mlflow"]["runs"]
        raise Exception("Unknown manifest format")

    def export_experiment(self, exp_id_or_name, output_dir, run_ids=None):
        """
        :param exp_id_or_name: Experiment ID or name.
        :param output_dir: Output directory.
        :param run_ids: List of run IDs to export. If None export all run IDs.
        :return: Number of successful and number of failed runs.
        """
        exp = mlflow_utils.get_experiment(self.mlflow_client, exp_id_or_name)
        print(f"Exporting experiment '{exp.name}' (ID {exp.experiment_id}) to '{output_dir}'")
        ok_run_ids = []
        failed_run_ids = []
        j = -1
        previous_ok_runs = self._get_previous_ok_runs(output_dir)
        if run_ids:
            for j,run_id in enumerate(run_ids):
                run = self.mlflow_client.get_run(run_id)
                self._export_run(j, run, output_dir, ok_run_ids, failed_run_ids, previous_ok_runs)
        else:
            for j,run in enumerate(SearchRunsIterator(self.mlflow_client, exp.experiment_id)):
                self._export_run(j, run, output_dir, ok_run_ids, failed_run_ids, previous_ok_runs)

        info_attr = {
            "num_total_runs": (j+1),
            "num_ok_runs": len(ok_run_ids),
            "num_failed_runs": len(failed_run_ids),
            "failed_runs": failed_run_ids
        }
        exp_dct = utils.strip_underscores(exp) 
        exp_dct["tags"] = dict(sorted(exp_dct["tags"].items()))

        mlflow_attr = { "experiment": exp_dct , "runs": ok_run_ids }
        io_utils.write_export_file(output_dir, "experiment.json", __file__, mlflow_attr, info_attr)

        msg = f"for experiment '{exp.name}' (ID: {exp.experiment_id})"
        if len(failed_run_ids) == 0:
            print(f"All {len(ok_run_ids)} runs succesfully exported {msg}")
        else:
            print(f"{len(ok_run_ids)/j} runs succesfully exported {msg}")
            print(f"{len(failed_run_ids)/j} runs failed {msg}")
        return len(ok_run_ids), len(failed_run_ids) 


    def _export_run(self, idx, run, output_dir, ok_run_ids, failed_run_ids, previous_ok_runs):
        run_dir = os.path.join(output_dir, run.info.run_id)
        if self.skip_previous_ok_runs and run.info.run_id in previous_ok_runs:
            print(f"Skipping run {idx+1}: {run.info.run_id}")
            res = True
        else:
            print(f"Exporting run {idx+1}: {run.info.run_id}")
            res = self.run_exporter.export_run(run.info.run_id, run_dir)
        if res:
            ok_run_ids.append(run.info.run_id)
        else:
            failed_run_ids.append(run.info.run_id)


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
    exporter.export_experiment(experiment, output_dir)

if __name__ == "__main__":
    main()
