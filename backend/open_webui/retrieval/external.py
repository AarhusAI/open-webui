# --- BEGIN EXTERNAL RETRIEVAL PATCH ---
# External retrieval engine: delegates document search to an external HTTP service
# instead of querying the built-in vector DB directly.
# --- END EXTERNAL RETRIEVAL PATCH ---

import logging
from typing import Optional, List

import requests

from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, REQUESTS_VERIFY
from open_webui.utils.headers import include_user_info_headers

log = logging.getLogger(__name__)


def query_external_retrieval(
    url: str,
    api_key: str,
    queries: List[str],
    collection_names: List[str],
    k: int,
    timeout: Optional[str] = None,
    user=None,
    messages: Optional[List[dict]] = None,
) -> Optional[dict]:
    """
    Query an external retrieval service.

    POST {url}/search with queries + collection_names + k.
    Optionally includes messages for external services that support message passthrough.
    Returns dict with keys: documents, metadatas, distances (matching internal format).
    Returns None on error.
    """
    payload = {
        "queries": queries,
        "collection_names": collection_names,
        "k": k,
    }

    if messages is not None:
        payload["messages"] = messages

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        request_timeout = int(timeout) if timeout else None

        log.info(
            f"query_external_retrieval: url={url}, queries={queries}, "
            f"messages={len(messages) if messages else 0}, "
            f"collections={collection_names}, k={k}"
        )

        r = requests.post(
            f"{url}/search",
            headers=headers,
            json=payload,
            timeout=request_timeout,
            verify=REQUESTS_VERIFY,
        )
        r.raise_for_status()
        data = r.json()

        if "documents" in data:
            return data
        else:
            log.error("No documents found in external retrieval response")
            return None

    except Exception as e:
        log.exception(f"Error in external retrieval: {e}")
        return None
