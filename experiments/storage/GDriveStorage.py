from __future__ import print_function

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from experiments.storage.Storage import Storage
from experiments.datasets.Base import Dataset
from experiments.utils import CustomJSONEncoder
from budgetsvm.svm import SVC


class GDriveStorage(Storage):
    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    CREDENTIALS_PATH: Path = Path.home() / "gdrive_budgetsvm_credentials.json"
    TOKEN_PATH: Path = Path.home() / "gdrive_budgetsvm_token.json"
    FOLDERS = ["models", "datasets", "results", "logs"]

    credentials: Credentials = None
    service: Resource = None
    folder_id_map: dict[str, str] = {}

    def __init__(self):
        self.__authenticate()
        self.service = build("drive", "v3", credentials=self.credentials)
        self.__init_gdrive_folder_ids()

    def __authenticate(self):
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time.
        if self.TOKEN_PATH.exists():
            self.credentials = Credentials.from_authorized_user_file(self.TOKEN_PATH, self.SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.CREDENTIALS_PATH, self.SCOPES)
                self.credentials = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.TOKEN_PATH, "w") as token:
                token.write(self.credentials.to_json())

    def __init_gdrive_folder_ids(self):
        for folder in self.FOLDERS:
            resp = (
                self.service.files()
                .list(
                    q=f"name='{folder}' and mimeType='application/vnd.google-apps.folder'",
                    spaces="drive",
                )
                .execute()
            )

            if len(resp["files"]):
                # folder already exists in gdrive
                self.folder_id_map[folder] = resp["files"][0]["id"]
            else:
                # create folder
                file = (
                    self.service.files()
                    .create(
                        body={
                            "name": folder,
                            "mimeType": "application/vnd.google-apps.folder",
                        },
                        fields="id",
                    )
                    .execute()
                )

                self.folder_id_map[folder] = file.get("id")

    def __upload_file(self, filepath: Path, parent_folder_name: str) -> bool:
        """Uploads a file to a gdrive folder. Returns True on success, false on failure."""
        file_metadata = {
            "name": filepath.name,
            "parents": [self.folder_id_map[parent_folder_name]],
        }
        chunk_size = 10 * 1024 * 1024  # 10 MB
        media = MediaFileUpload(filepath, chunksize=chunk_size, resumable=True)

        try:
            request = self.service.files().create(body=file_metadata, media_body=media, fields="id")
            upload_response = None
            while upload_response is None:
                _, upload_response = request.next_chunk()

            # upload outcome
            return "id" in upload_response
        except HttpError:
            return False

    def save_model(self, model: SVC, model_id: str):
        logging.info(f"saving model {model_id} on gdrive")

        tmp_filepath = Path(f"{model_id}.pkl")

        with open(tmp_filepath, "wb") as f:
            pickle.dump(model, f)

        success = self.__upload_file(tmp_filepath, "models")

        if not success:
            logging.error(f"upload of model {model_id} failed! Keeping file locally.")
            return

        tmp_filepath.unlink()

    def save_dataset(self, ds: Dataset):
        logging.info(f"saving dataset {ds.id} on gdrive")

        tmp_filepath = Path(f"{ds.id}.json")

        with open(tmp_filepath, "w") as f:
            json.dump(ds, f, cls=CustomJSONEncoder)

        success = self.__upload_file(tmp_filepath, "datasets")

        if not success:
            logging.error(f"upload of dataset {ds.id} failed! Keeping file locally.")
            return

        tmp_filepath.unlink()

    def save_results(self, res: list[dict], timestamp: str):
        logging.info(f"saving results of timestamp {timestamp} on gdrive")

        tmp_filepath = Path(f"{timestamp}.json")
        with open(tmp_filepath, "w") as f:
            f.write(json.dumps(res, cls=CustomJSONEncoder))

        success = self.__upload_file(tmp_filepath, "results")
        if not success:
            logging.error(f"upload of results {timestamp} failed! Keeping file locally.")
            return

        tmp_filepath.unlink()

    def save_log(self, log_file_path: Path, timestamp: str):
        logging.info(f"saving logs for experiment {timestamp} on gdrive")

        success = self.__upload_file(log_file_path, "logs")
        if not success:
            logging.error(f"upload of logs for experiment {timestamp} failed! Keeping file locally.")
            return

        log_file_path.unlink()

    def get_dataset_if_exists(self, dataset_hash: str) -> Optional[Dataset]:
        # search for the file
        resp = (
            self.service.files()
            .list(
                q=f"'{self.folder_id_map['datasets']}' in parents and name='{dataset_hash}.json'  and trashed = false",
                spaces="drive",
                fields="files(id)",
            )
            .execute()
        )

        if len(resp["files"]) == 0:
            return None

        file_id = resp["files"].pop()["id"]

        request = self.service.files().get_media(fileId=file_id)
        tmp_filepath = Path(f"{dataset_hash}.json")
        with open(tmp_filepath, "wb") as file:
            downloader = MediaIoBaseDownload(file, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

        with open(tmp_filepath, "r") as f:
            ds = json.load(f)

        tmp_filepath.unlink()

        return Dataset.from_json(ds)

    def list_results_files(self):
        resp = (
            self.service.files()
            .list(
                q=f"'{self.folder_id_map['results']}' in parents and trashed=false",
                spaces="drive",
                fields="files(name)",
            )
            .execute()
        )

        if len(resp["files"]) == 0:
            return None

        for filename in sorted(resp["files"], key=lambda item: item["name"]):
            filename = filename["name"].split("_")[0].split(".json")[0]

            readable_date = datetime.fromtimestamp(float(filename)).strftime("%d/%m/%Y %H:%M")
            print(f"{readable_date}\t{filename}.json")

    def get_results_no_schema(self, target_file_name):
        file_list = (
            self.service.files()
            .list(
                q=f"'{self.folder_id_map['results']}' in parents and trashed=false",
                spaces="drive",
                fields="files(id,name)",
            )
            .execute()
        )
        file = next(file for file in file_list["files"] if file["name"] == target_file_name)

        request = self.service.files().get_media(fileId=file["id"])
        tmp_filepath = Path(f"{file['id']}.json")
        with open(tmp_filepath, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

        with open(tmp_filepath, "r") as f:
            df = pd.read_json(f)

        tmp_filepath.unlink()
        return df

    def get_result_dataframe(self, target_file_name):
        file_list = (
            self.service.files()
            .list(
                q=f"'{self.folder_id_map['results']}' in parents and trashed=false",
                spaces="drive",
                fields="files(id,name)",
            )
            .execute()
        )
        file = next(file for file in file_list["files"] if file["name"] == target_file_name)

        request = self.service.files().get_media(fileId=file["id"])
        tmp_filepath = Path(f"{file['id']}.json")
        with open(tmp_filepath, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

        with open(tmp_filepath, "r") as f:
            df = pd.read_json(f)

        tmp_filepath.unlink()

        # add budget percentage as column
        df["budget_percentage"] = df["model_name"].apply(lambda x: float(x.split("_")[0]) if "full" not in x else 1.0)

        # convert training time in minutes
        df.train_time = df.train_time.div(60)
        df = df.rename({"train_time": "train_time_min"}, axis=1)

        # set datatypes
        df = df.astype(
            {
                "dataset": "string",
                "model_UUID": "string",
                "model_name": "string",
                "solver_status": "int",
                "params": "object",
                "score": "float",
                "budget": "float",
                "num_sv": "float",
                "train_time_min": "float",
                "s_pos": "int",
                "s_neg": "int",
                "b_pos": "int",
                "b_neg": "int",
                "mip_gap": "float",
            }
        )

        full_budget_score = df.query("budget==inf")[["dataset", "score"]]
        df = df.join(full_budget_score.set_index("dataset"), on="dataset", rsuffix="_full_budget")
        df["score_ratio"] = df["score"] / df["score_full_budget"]
        return df


if __name__ == "__main__":
    storage = GDriveStorage()
    storage.get_result_dataframe("1689153514.3068242.json")
