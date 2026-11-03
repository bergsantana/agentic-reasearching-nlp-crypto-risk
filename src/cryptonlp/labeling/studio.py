"""A small Label Studio REST client: only the calls the labeling workflow makes.

Authentication uses a personal access token (Label Studio 1.23 disables legacy tokens by
default): the token is a JWT refresh token, exchanged at ``/api/token/refresh/`` for an access
token that lasts about five minutes. The client exchanges it again when it expires.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import httpx

# Exchange the token again this long before the access token expires.
_ACCESS_MARGIN_S = 30.0
_ACCESS_LIFETIME_S = 240.0  # Label Studio issues 300 s access tokens


class StudioError(RuntimeError):
    pass


class Studio:
    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise StudioError(
                "no Label Studio token: create a personal access token in Label Studio "
                "(Account & Settings) and set CRYPTONLP_LABEL_STUDIO_API_KEY in .env"
            )
        self._refresh = api_key
        self._access: str | None = None
        self._expires = 0.0
        self._http = httpx.Client(base_url=url.rstrip("/"), transport=transport, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Studio:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- plumbing ---------------------------------------------------------------------------

    def _token(self, *, renew: bool = False) -> str:
        if renew or self._access is None or time.monotonic() > self._expires - _ACCESS_MARGIN_S:
            r = self._http.post("/api/token/refresh/", json={"refresh": self._refresh})
            if r.status_code != 200:
                raise StudioError(
                    f"Label Studio refused the token ({r.status_code}): is "
                    "CRYPTONLP_LABEL_STUDIO_API_KEY a personal access token from this instance?"
                )
            self._access = cast(str, r.json()["access"])
            self._expires = time.monotonic() + _ACCESS_LIFETIME_S
        assert self._access is not None
        return self._access

    def _call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in (0, 1):
            headers = {"Authorization": f"Bearer {self._token(renew=attempt == 1)}"}
            r = self._http.request(method, path, headers=headers, **kwargs)
            if r.status_code != 401:
                break
        if r.status_code >= 400 and not (method == "DELETE" and r.status_code == 404):
            raise StudioError(f"{method} {path}: {r.status_code} {r.text[:300]}")
        return r

    # --- calls ------------------------------------------------------------------------------

    def validate_config(self, label_config: str) -> None:
        self._call("POST", "/api/projects/validate", json={"label_config": label_config})

    def create_project(self, title: str, label_config: str, description: str = "") -> int:
        r = self._call(
            "POST",
            "/api/projects",
            json={
                "title": title,
                "description": description,
                "label_config": label_config,
                # One annotation per task, in the order imported (already shuffled), no skips,
                # no empty submissions.
                "maximum_annotations": 1,
                "sampling": "Sequential sampling",
                "show_skip_button": False,
                "enable_empty_annotation": False,
                "show_instruction": False,
            },
        )
        return cast(int, r.json()["id"])

    def import_tasks(self, project_id: int, tasks: Sequence[dict[str, Any]]) -> None:
        self._call("POST", f"/api/projects/{project_id}/import", json=list(tasks))

    def task_ids(self, project_id: int) -> dict[int, int]:
        """``item_id`` (ours, in the task data) to Label Studio task ID, for every task."""
        out: dict[int, int] = {}
        page = 1
        while True:
            r = self._call(
                "GET",
                "/api/tasks",
                params={"project": project_id, "page": page, "page_size": 500, "fields": "all"},
            )
            body = r.json()
            tasks = body["tasks"] if isinstance(body, dict) else body
            for t in tasks:
                out[int(t["data"]["item_id"])] = int(t["id"])
            total = body.get("total", len(out)) if isinstance(body, dict) else len(out)
            if not tasks or len(out) >= total:
                return out
            page += 1

    def export(self, project_id: int) -> list[dict[str, Any]]:
        """Every task of the project with its annotations."""
        r = self._call(
            "GET",
            f"/api/projects/{project_id}/export",
            params={"exportType": "JSON", "download_all_tasks": "true"},
        )
        return cast(list[dict[str, Any]], r.json())

    def delete_task(self, task_id: int) -> None:
        self._call("DELETE", f"/api/tasks/{task_id}/")
