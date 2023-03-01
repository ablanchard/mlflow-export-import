"""
Import a registered model and all the experiment runs associated with its latest versions.
"""

import os
import click

import mlflow
from mlflow.exceptions import RestException, MlflowException

from mlflow_export_import.common.click_options import opt_input_dir, opt_model, \
  opt_experiment_name, opt_delete_model, opt_import_source_tags, opt_verbose
from mlflow_export_import.common import io_utils
from mlflow_export_import.common import model_utils
from mlflow_export_import.common.source_tags import set_source_tags_for_field, fmt_timestamps
from mlflow_export_import.common import MlflowExportImportException
from mlflow_export_import.run.import_run import RunImporter



def _set_source_tags_for_field(dct, tags):
    set_source_tags_for_field(dct, tags)
    fmt_timestamps("creation_timestamp", dct, tags)
    fmt_timestamps("last_updated_timestamp", dct, tags)


class BaseModelImporter():
    """ Base class of ModelImporter subclasses. """

    def __init__(self, mlflow_client, run_importer=None, import_source_tags=False, await_creation_for=None):
        """
        :param mlflow_client: MLflow client or if None create default client.
        :param run_importer: RunImporter instance.
        :param import_source_tags: Import source information for MLFlow objects and create tags in destination object.
        :param await_creation_for: Seconds to wait for model version crreation.
        """
        self.mlflow_client = mlflow_client 
        self.run_importer = run_importer if run_importer else RunImporter(self.mlflow_client, import_source_tags=import_source_tags, mlmodel_fix=True)
        self.import_source_tags = import_source_tags 
        self.await_creation_for = await_creation_for 


    def _import_version(self, model_name, src_vr, dst_run_id, dst_source, sleep_time):
        """
        :param model_name: Model name.
        :param src_vr: Source model version.
        :param dst_run: Destination run.
        :param dst_source: Destination version 'source' field.
        :param sleep_time: Seconds to wait for model version crreation.
        """
        dst_source = dst_source.replace("file://","") # OSS MLflow
        if not dst_source.startswith("dbfs:") and not os.path.exists(dst_source):
            print(f"'source' argument for MLflowClient.create_model_version does not exist: {dst_source}")
            raise MlflowExportImportException(f"'source' argument for MLflowClient.create_model_version does not exist: {dst_source}")
        tags = src_vr["tags"]
        if self.import_source_tags:
            _set_source_tags_for_field(src_vr, tags)
        try:
            version = self.mlflow_client.create_model_version(model_name, dst_source, dst_run_id, \
                description=src_vr["description"], tags=tags, await_creation_for=0)
        except MlflowException as e:
            if "Max retries" in e.message:
                print("Hitting max retiries, Waiting for model version creation to complete...")
                model_utils.wait_until_version_is_ready(self.mlflow_client, model_name, int(src_vr["version"]) - 1, sleep_time=1, iterations=300)
                version = self.mlflow_client.create_model_version(model_name, dst_source, dst_run_id, \
                    description=src_vr["description"], tags=tags, await_creation_for=0)
        return version

    def _import_model(self, model_name, input_dir, delete_model=False):
        """
        :param model_name: Model name.
        :param input_dir: Input directory.
        :param delete_model: Delete current model before importing versions.
        :param verbose: Verbose.
        :param sleep_time: Seconds to wait for model version crreation.
        :return: Model import manifest.
        """
        path = os.path.join(input_dir, "model.json")
        model_dct = io_utils.read_file_mlflow(path)["registered_model"]

        # print("Model to import:")
        print(f"  Name: {model_dct['name']}")
        # print(f"  Description: {model_dct.get('description','')}")
        # print(f"  Tags: {model_dct.get('tags','')}")
        # print(f"  {len(model_dct['latest_versions'])} latest versions")
        # print(f"  path: {path}")

        if not model_name:
            model_name = model_dct["name"]
        if delete_model:
            model_utils.delete_model(self.mlflow_client, model_name)
        else:
            print("NOT deleting existing model")

        try:
            tags = { e["key"]:e["value"] for e in model_dct.get("tags", {}) }
            if self.import_source_tags:
                _set_source_tags_for_field(model_dct, tags)
            self.mlflow_client.create_registered_model(model_name, tags, model_dct.get("description"))
            print(f"Created new registered model '{model_name}'")
        except RestException as e:
            if not "RESOURCE_ALREADY_EXISTS: Registered Model" in str(e):
                raise e
            print(f"Registered model '{model_name}' already exists")
            # Updating tags
            for key, value in tags.items():
                print("Updating tag", key, value)
                self.mlflow_client.set_registered_model_tag(model_name, key, value)
        return model_dct


class ModelImporter(BaseModelImporter):
    """ Low-level 'point' model importer.  """

    def __init__(self, mlflow_client, run_importer=None, import_source_tags=False, await_creation_for=None):
        super().__init__(mlflow_client, run_importer, import_source_tags=import_source_tags, await_creation_for=await_creation_for)


    def import_model(self, model_name, input_dir, experiment_name, delete_model=False, verbose=False, sleep_time=30):
        """
        :param model_name: Model name.
        :param input_dir: Input directory.
        :param experiment_name: The name of the experiment.
        :param delete_model: Delete current model before importing versions.
        :param import_source_tags: Import source information for registered model and its versions ad tags in destination object.
        :param verbose: Verbose.
        :param sleep_time: Seconds to wait for model version crreation.
        :return: Model import manifest.
        """
        model_dct = self._import_model(model_name, input_dir, delete_model)
        mlflow.set_experiment(experiment_name)
        print("Importing versions:")
        for vr in model_dct["latest_versions"]:
            run_id = self._import_run(input_dir, experiment_name, vr)
            self.import_version(model_name, vr, run_id, sleep_time)
        if verbose:
            model_utils.dump_model_versions(self.mlflow_client, model_name)


    def _import_run(self, input_dir, experiment_name, vr):
        run_id = vr["run_id"]
        source = vr["source"]
        current_stage = vr["current_stage"]
        run_artifact_uri = vr.get("_run_artifact_uri",None)
        run_dir = os.path.join(input_dir,run_id)
        print(f"  Version {vr['version']}:")
        print(f"    current_stage: {current_stage}:")
        print(f"    Source run - run to import:")
        print(f"      run_id: {run_id}")
        print(f"      run_artifact_uri: {run_artifact_uri}")
        print(f"      source:           {source}")
        model_path = _extract_model_path(source, run_id)
        print(f"      model_path:   {model_path}")
        dst_run,_ = self.run_importer.import_run(experiment_name, run_dir)
        dst_run_id = dst_run.info.run_id
        run = self.mlflow_client.get_run(dst_run_id)
        print(f"    Destination run - imported run:")
        print(f"      run_id: {dst_run_id}")
        print(f"      run_artifact_uri: {run.info.artifact_uri}")
        source = _path_join(run.info.artifact_uri, model_path)
        print(f"      source:           {source}")
        return dst_run_id


    def import_version(self, model_name, src_vr, dst_run_id, sleep_time):
        dst_run = self.mlflow_client.get_run(dst_run_id)
        model_path = _extract_model_path(src_vr["source"], src_vr["run_id"])
        dst_source = f"{dst_run.info.artifact_uri}/{model_path}"
        self._import_version(model_name, src_vr, dst_run_id, dst_source, sleep_time)


class AllModelImporter(BaseModelImporter):
    """ High-level 'bulk' model importer.  """

    def __init__(self, mlflow_client, run_info_map, run_importer=None, import_source_tags=False, await_creation_for=None, conf={}):
        super().__init__(mlflow_client, run_importer, import_source_tags=import_source_tags, await_creation_for=await_creation_for)
        self.run_info_map = run_info_map
        self.conf = conf

    
    def fix_missing_version(self, versions):
        import copy
        versions_int = [int(v["version"]) for v in versions]
        # find which version is missing in range
        missing_versions = [v for v in range(1, max(versions_int)+1) if v not in versions_int]
        results = []
        if len(missing_versions) == 0:
            print("No missing version")
        for missing in missing_versions:
            if missing == 1:
                # take number first than exist
                ordered_versions = sorted(versions, key=lambda x: int(x["version"]))
                added_version = copy.deepcopy(ordered_versions[0])
                added_version["description"] = f"Copy of version {added_version['version']} fixed to not mess with the version ordering"
                added_version["version"] = str(missing)
                added_version["current_stage"] = "Archived"
                results.append(added_version)
            else:
                # take number before
                previous_version = [v for v in versions if int(v["version"]) == missing-1]
                if len(previous_version) == 0:
                    previous_version = [v for v in results if int(v["version"]) == missing-1]
                added_version = copy.deepcopy(previous_version[0])
                added_version["description"] = f"Copy of version {added_version['version']} fixed to not mess with the version ordering"
                added_version["version"] = str(missing)
                added_version["current_stage"] = "Archived"
                results.append(added_version)
        print(f"Added missing versions: {results}")
        return results

    def fix_missing_run_version(self, versions, model_name):
        if "missing_runs" not in self.conf:
            return
        
        missing_runs = self.conf["missing_runs"]
        if model_name not in missing_runs:
            print("nothing to fix")
            return None
        missing_run_ids = missing_runs[model_name]
        for id in missing_run_ids:
            # find version with id
            version_to_fix = [v for v in versions if v["run_id"] == id][0]
            version_to_copy = [v for v in versions if int(v["version"]) == int(version_to_fix["version"])-1][0]
            version_to_fix["description"] = f"Copy of version {version_to_copy['version']} fixed to not mess with the version ordering"
            version_to_fix["current_stage"] = "Archived"
            version_to_fix["run_id"] = version_to_copy["run_id"]
            version_to_fix["source"] = version_to_copy["source"]
            version_to_fix["user_id"] = version_to_copy["user_id"]
            version_to_fix["_run_artifact_uri"] = version_to_copy["_run_artifact_uri"]
            print(f"Fixed {version_to_fix['version']} with {version_to_copy['version']} new run_id: {version_to_fix['run_id']}")

    def remove_version(self, versions):
        if "missing_experiments" not in self.conf:
            return
        missing_experiments = self.conf["missing_experiments"]
        for version in versions:
            if version["_experiment_name"] in missing_experiments:
                versions.remove(version)


    def replace_if_user_is_archived(self, exp_name):
        if "archived_users" not in self.conf:
            return exp_name
        archived_users = self.conf["archived_users"]
        for user in archived_users:
            if exp_name.startswith(f"/Users/{user}"):
                return exp_name.replace("/Users/", "/Archive/")
        return exp_name

    def import_model(self, model_name, input_dir, delete_model=False, verbose=False, sleep_time=3):
        """
        :param model_name: Model name.
        :param input_dir: Input directory.
        :param delete_model: Delete current model before importing versions.
        :param verbose: Verbose.
        :param sleep_time: Seconds to wait for model version crreation.
        :return: Model import manifest.
        """
        try:

            if os.path.exists(os.path.join(input_dir, "import-model.json")):
                previous_versions = io_utils.read_file(os.path.join(input_dir, "import-model.json"))
            else:
                previous_versions = []
            model_dct = self._import_model(model_name, input_dir, len(previous_versions) == 0)
            # order latest versions by version
            all = model_dct["latest_versions"]
            self.remove_version(all)
            all += self.fix_missing_version(all)
            self.fix_missing_run_version(all, model_name)
            todo = list(filter(lambda x: x["version"] not in previous_versions, all))
            versions = sorted(todo, key=lambda x: int(x["version"]))

            print(f"Importing {len(all)} latest versions, skipping {len(previous_versions)} previous versions")

            for vr in versions:
                print(f"Doing {vr['run_id']} version {vr['version']}")
                src_run_id = vr["run_id"]
                dst_run_id = self.run_info_map[src_run_id]["dst_run_id"]
                mlflow.set_experiment(self.replace_if_user_is_archived(vr["_experiment_name"]))
                created_version = self.import_version(model_name, vr, dst_run_id, sleep_time)
                if created_version.version != vr["version"]:
                    print(f"Version mismatch: Source version {vr['version']} was created as version: {created_version.version}")
                    raise Exception(f"Version mismatch: Source version {vr['version']} was created as version: {created_version.version}")
            
            # transition version to stage
            # We need to do it also for previous version because stage could have changed between 2 exports
            to_transition = sorted(all, key=lambda x: int(x["version"]))
            for vr in to_transition:
                if vr["current_stage"] != "None":
                    print(f"Transitioning version {vr['version']} to stage {vr['current_stage']}")
                    model_utils.wait_until_version_is_ready(self.mlflow_client, model_name, vr['version'], sleep_time=1, iterations=300)
                    try:
                        self.mlflow_client.transition_model_version_stage(model_name, vr['version'], vr["current_stage"])
                    except Exception as e:
                        if "Cannot update model version to its current stage" not in str(e):
                            raise e
            
            

            if verbose:
                model_utils.dump_model_versions(self.mlflow_client, model_name)

            # create list with version from all
            to_write = [x["version"] for x in all]
            io_utils.write_file(os.path.join(input_dir, "import-model.json"), to_write)
            print(f"Success done importing model {model_name}")

            
        except Exception as e:
            print(f"Error importing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def import_version(self, model_name, src_vr, dst_run_id, sleep_time):
        src_run_id = src_vr["run_id"]
        model_path = _extract_model_path(src_vr["source"], src_run_id)
        dst_artifact_uri = self.run_info_map[src_run_id]["artifact_uri"]
        dst_source = f"{dst_artifact_uri}/{model_path}"
        return self._import_version(model_name, src_vr, dst_run_id, dst_source, sleep_time)


def _extract_model_path(source, run_id):
    idx = source.find(run_id)
    model_path = source[1+idx+len(run_id):]
    if model_path.startswith("artifacts/"): # Bizarre - sometimes there is no 'artifacts' after run_id
        model_path = model_path.replace("artifacts/","")
    return model_path


def _path_join(x,y):
    """ Account for DOS backslash """
    path = os.path.join(x,y)
    if path.startswith("dbfs:"):
        path = path.replace("\\","/") 
    return path


@click.command()
@opt_input_dir
@opt_model
@opt_experiment_name
@opt_delete_model
@opt_import_source_tags
@click.option("--await-creation-for",
    help="Await creation for specified seconds.",
    type=int,
    default=None,
    show_default=True
)
@click.option("--sleep-time",
    help="Sleep time for polling until version.status==READY.",
    type=int,
    default=5,
)
@opt_verbose
def main(input_dir, model, experiment_name, delete_model, await_creation_for, import_source_tags, verbose, sleep_time):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    client = mlflow.client.MlflowClient()
    importer = ModelImporter(client, import_source_tags=import_source_tags, await_creation_for=await_creation_for)
    importer.import_model(model, input_dir, experiment_name, delete_model, verbose, sleep_time)


if __name__ == "__main__":
    main()
