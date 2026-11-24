import hashlib
import json
import mimetypes
import tempfile
import threading
import time
from functools import partial
from pathlib import Path

import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httplib2 import HttpLib2Error

from config import load_config, log

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_TYPE = "application/vnd.google-apps.folder"
MANIFEST = "manifest.json"
CHUNK = 64 << 20
CHUNK_RETRIES = 3
ATTEMPTS = 5
BACKOFF_SECONDS = 10
NO_CREDENTIALS = "No Google credentials on this VM. Run `colab auth -s <session>` and approve, then rerun."


class DriveMismatch(Exception):
    pass


RETRYABLE = (HttpError, HttpLib2Error, OSError, TransportError, DriveMismatch)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def drive_service():
    credentials, _ = google.auth.default(scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def check_drive_access():
    try:
        drive_service().about().get(fields="user").execute()
    except (DefaultCredentialsError, RefreshError, TransportError, HttpError) as error:
        raise SystemExit(f"{NO_CREDENTIALS} ({type(error).__name__})")


def drive_folder_names(run_dir):
    return run_dir.removeprefix("/content/drive/").removeprefix("MyDrive/").strip("/").split("/")


def local_stage_dir(names):
    base = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    stage = base.joinpath("looped-stage", *names)
    stage.mkdir(parents=True, exist_ok=True)
    return stage


class DriveFolder:
    def __init__(self, config):
        self.config = config
        self.names = drive_folder_names(config["run_dir"])
        self.stage = local_stage_dir(self.names)
        self.thread_state = threading.local()
        self.manifest_lock = threading.Lock()
        self.file_ids = {}
        self.folder_id = self.with_retries("find the run folder", self.find_run_folder)
        self.manifest = self.read_manifest()

    def service(self):
        if not hasattr(self.thread_state, "service"):
            self.thread_state.service = drive_service()
        return self.thread_state.service

    def with_retries(self, what, action):
        for attempt in range(1, ATTEMPTS + 1):
            try:
                return action()
            except RETRYABLE as error:
                if attempt == ATTEMPTS:
                    raise
                wait = BACKOFF_SECONDS * 2 ** (attempt - 1)
                log(self.config, "storage", f"{what} failed ({type(error).__name__}: {error}), retrying in {wait}s")
                time.sleep(wait)

    def find_run_folder(self):
        parent = "root"
        for name in self.names:
            parent = self.child_folder(parent, name)
        return parent

    def child_folder(self, parent, name):
        query = f"mimeType = '{FOLDER_TYPE}' and name = '{name}' and '{parent}' in parents and trashed = false"
        found = self.service().files().list(q=query, fields="files(id)", orderBy="createdTime").execute()["files"]
        if found:
            return found[0]["id"]
        body = {"name": name, "mimeType": FOLDER_TYPE, "parents": [parent]}
        return self.service().files().create(body=body, fields="id").execute()["id"]

    def file_id(self, name):
        if name not in self.file_ids:
            query = f"name = '{name}' and '{self.folder_id}' in parents and trashed = false"
            listing = self.service().files().list(q=query, fields="files(id)", orderBy="modifiedTime desc")
            found = self.with_retries(f"find {name}", listing.execute)["files"]
            if found:
                self.file_ids[name] = found[0]["id"]
        return self.file_ids.get(name)

    def metadata(self, file_id):
        return self.service().files().get(fileId=file_id, fields="md5Checksum, size").execute()

    def read_manifest(self):
        file_id = self.file_id(MANIFEST)
        if file_id is None:
            return {}
        request = self.service().files().get_media(fileId=file_id)
        return json.loads(self.with_retries(f"read {MANIFEST}", request.execute))

    def local(self, name):
        return self.stage / name

    def send(self, name, path):
        mimetype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        media = MediaFileUpload(str(path), mimetype=mimetype, chunksize=CHUNK, resumable=True)
        file_id = self.file_id(name)
        files = self.service().files()
        if file_id is None:
            request = files.create(body={"name": name, "parents": [self.folder_id]}, media_body=media, fields="id")
        else:
            request = files.update(fileId=file_id, media_body=media, fields="id")
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=CHUNK_RETRIES)
        self.file_ids[name] = response["id"]
        return response["id"]

    def send_verified(self, name, path, local_md5, size):
        file_id = self.send(name, path)
        remote = self.metadata(file_id)
        if remote.get("md5Checksum") != local_md5 or int(remote["size"]) != size:
            raise DriveMismatch(f"{name} on Drive is {remote}, local is md5 {local_md5} size {size}")
        return file_id

    def drop_old_revisions(self, file_id):
        head = self.service().files().get(fileId=file_id, fields="headRevisionId").execute()["headRevisionId"]
        listing = self.service().revisions().list(fileId=file_id, fields="revisions(id)").execute()
        for revision in listing["revisions"]:
            if revision["id"] != head:
                self.service().revisions().delete(fileId=file_id, revisionId=revision["id"]).execute()

    def upload(self, name):
        path = self.local(name)
        size, local_md5 = path.stat().st_size, md5(path)
        file_id = self.with_retries(f"upload {name}", partial(self.send_verified, name, path, local_md5, size))
        self.with_retries(f"drop old {name} revisions", partial(self.drop_old_revisions, file_id))
        return {"bytes": size, "md5": local_md5, "id": file_id, "written": utc_now()}

    def save(self, name):
        record = self.upload(name)
        with self.manifest_lock:
            self.manifest[name] = record
            self.local(MANIFEST).write_text(json.dumps(self.manifest, indent=1))
            self.upload(MANIFEST)
        log(self.config, "storage", f"saved {name}: {record['bytes'] / 2**30:.2f} GiB, md5 {record['md5']}")
        return record["md5"]

    def save_text(self, name, text):
        self.local(name).write_text(text)
        return self.save(name)

    def fetch(self, file_id, path):
        with open(path, "wb") as file:
            downloader = MediaIoBaseDownload(file, self.service().files().get_media(fileId=file_id), chunksize=CHUNK)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=CHUNK_RETRIES)

    def fetch_verified(self, name, file_id, remote_md5):
        path = self.local(name)
        if path.exists() and md5(path) == remote_md5:
            return path
        self.fetch(file_id, path)
        if md5(path) != remote_md5:
            raise DriveMismatch(f"downloaded {name} does not match Drive md5 {remote_md5}")
        return path

    def download(self, name):
        file_id = self.file_id(name)
        if file_id is None:
            return None
        remote_md5 = self.with_retries(f"check {name}", partial(self.metadata, file_id))["md5Checksum"]
        recorded = self.manifest.get(name)
        if recorded is not None and recorded["md5"] != remote_md5:
            raise RuntimeError(f"{name} on Drive is md5 {remote_md5}, {MANIFEST} says {recorded['md5']}")
        path = self.with_retries(f"download {name}", partial(self.fetch_verified, name, file_id, remote_md5))
        log(self.config, "storage", f"downloaded {name}: md5 {remote_md5}")
        return path

    def verify_all(self):
        failures = []
        for name, record in self.manifest.items():
            remote = self.with_retries(f"check {name}", partial(self.metadata, record["id"]))
            if remote.get("md5Checksum") != record["md5"] or int(remote["size"]) != record["bytes"]:
                failures.append(f"{name} on Drive is {remote}, {MANIFEST} says md5 {record['md5']} size {record['bytes']}")
            else:
                log(self.config, "verify", f"{name} ok ({record['bytes'] / 2**30:.2f} GiB, md5 {record['md5']})")
        if failures:
            raise SystemExit("verification failed: " + "; ".join(failures))
        log(self.config, "verify", f"all {len(self.manifest)} files match {MANIFEST} on Drive")


if __name__ == "__main__":
    DriveFolder(load_config()).verify_all()
