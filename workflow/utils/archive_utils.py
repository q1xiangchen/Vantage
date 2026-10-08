"""Archive a run's sessions/ dir into a single sessions.zip.

Motivation: each session dir costs ~4 inodes and a full run holds 1k-3k sessions,
which strains filesystems with inode quotas. One zip per run costs 1 inode
and stays randomly accessible (read a single member without extracting).

Zip layout mirrors the dir: member paths are relative to sessions/, e.g.
`session-<id>/trace.jsonl`.
"""
import os
import shutil
import zipfile
from typing import Optional


def archive_sessions(work_dir: str, remove: bool = True) -> Optional[str]:
    """Pack `<work_dir>/sessions/` into `<work_dir>/sessions.zip`, then delete the
    dir (unless `remove=False`). Returns the zip path, or None when there is no
    sessions dir to pack.

    Safe under --resume: an existing sessions.zip is merged in — a session that was
    re-run on disk supersedes ALL of its old zip entries (whole-session granularity,
    so a re-run never mixes old and new files). The zip is written to a temp name
    and verified before it replaces the old one, and the dir is only deleted after
    that, so a crash mid-archive never loses sessions.
    """
    sessions_dir = os.path.join(work_dir, 'sessions')
    if not os.path.isdir(sessions_dir):
        return None
    zip_path = os.path.join(work_dir, 'sessions.zip')
    tmp_path = zip_path + '.tmp'

    # Exclusive per-run lock: two archivers racing on one run once deleted each
    # other's sessions mid-walk and produced zips with silently missing members.
    # A SIGKILLed archiver leaves a stale lock; its content says what to remove.
    lock_path = zip_path + '.lock'
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f'{lock_path} exists — another archiver is packing this run '
            '(or died holding the lock; delete the lock file if so)')
    os.write(lock_fd, f'pid {os.getpid()}\n'.encode())
    try:
        on_disk_sessions = set(os.listdir(sessions_dir))
        added = set()
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for root, _, files in os.walk(sessions_dir):
                for fn in files:
                    path = os.path.join(root, fn)
                    arcname = os.path.relpath(path, sessions_dir)
                    zout.write(path, arcname)
                    added.add(arcname)
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path) as zin:
                    for info in zin.infolist():
                        top = info.filename.split('/', 1)[0]
                        if top in on_disk_sessions:
                            continue
                        zout.writestr(info, zin.read(info.filename))

        # The dir changing under us means a live run (or unlocked writer) is
        # touching it — the walk may have missed sessions, so keep the dir.
        if set(os.listdir(sessions_dir)) != on_disk_sessions:
            os.unlink(tmp_path)
            raise RuntimeError(
                f'{sessions_dir} changed while packing (live run?); aborted')

        with zipfile.ZipFile(tmp_path) as zf:
            names = set(zf.namelist())
            missing = added - names
            if missing or zf.testzip() is not None:
                raise RuntimeError(
                    f'sessions archive verification failed for {tmp_path} '
                    f'(missing={len(missing)}); sessions/ left untouched')
        os.replace(tmp_path, zip_path)

        if remove:
            shutil.rmtree(sessions_dir)
    finally:
        os.close(lock_fd)
        os.unlink(lock_path)
    return zip_path


def session_file_exists(work_dir: str, relpath: str) -> bool:
    """True when `sessions/<relpath>` exists in the live dir or in sessions.zip
    (without reading the file's content, unlike read_session_file)."""
    if os.path.exists(os.path.join(work_dir, 'sessions', relpath)):
        return True
    zip_path = os.path.join(work_dir, 'sessions.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            return relpath.replace(os.sep, '/') in zf.namelist()
    return False


def read_session_file(work_dir: str, relpath: str) -> Optional[bytes]:
    """Read `sessions/<relpath>` of a run, from the live dir if present, else from
    sessions.zip. Returns None when the file exists in neither."""
    path = os.path.join(work_dir, 'sessions', relpath)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    zip_path = os.path.join(work_dir, 'sessions.zip')
    if os.path.exists(zip_path):
        member = relpath.replace(os.sep, '/')
        with zipfile.ZipFile(zip_path) as zf:
            if member in zf.namelist():
                return zf.read(member)
    return None
