import time
from mlflow.exceptions import RestException
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow_export_import.common.timestamp_utils import fmt_ts_millis


def delete_model(client, model_name, sleep_time=5):
    """ Delete a model and all its versions. """
    versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    print(f"Deleting model '{model_name}' and {len(versions)} versions")
    for v in versions:
        print(f"  version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}")
        if v.current_stage != "Archived":
            client.transition_model_version_stage(model_name, v.version, "Archived")
    for v in versions:
        client.delete_model_version(model_name, v.version)
    client.delete_registered_model(model_name)
    print(f"Deleted model '{model_name}'")


def wait_until_version_is_ready(client, model_name, model_version, sleep_time=1, iterations=100):
    """ Due to blob eventual consistency, wait until a newly created version is in READY state. """
    start = time.time()
    status = ModelVersionStatus.PENDING_REGISTRATION
    while status != ModelVersionStatus.READY and time.time() - start <= iterations:
        vr = client.get_model_version(model_name, model_version.version)
        status = ModelVersionStatus.from_string(vr.status)
        print(f"Version: id={vr.version} status={vr.status} state={vr.current_stage}")
        time.sleep(sleep_time)
    end = time.time()
    print(f"Waited {round(end-start,2)} seconds")


def show_versions(model_name, versions, msg):
    """ Display as table registered model versions. """
    import pandas as pd
    from tabulate import tabulate
    versions = [ [vr.version, vr.current_stage, vr.status, vr.run_id,
           fmt_ts_millis(vr.creation_timestamp),
           fmt_ts_millis(vr.last_updated_timestamp),
           vr.description
       ] for vr in versions ]
    df = pd.DataFrame(versions, columns = [
        "version",
        "current_stage",
        "status", 
        "run_id", 
        "creation_timestamp",
        "last_updated_timestamp",
        "description"
    ])
    print(f"\n'{msg}' {len(versions)} versions for model '{model_name}'")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


def dump_model_versions(client, model_name):
    """ Display as table 'latest' and 'all' registered model versions. """
    versions = client.get_latest_versions(model_name)
    show_versions(model_name, versions, "Latest")
    versions = client.search_model_versions(f"name='{model_name}'")
    show_versions(model_name, versions, "All")
