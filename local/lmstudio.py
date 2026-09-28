"""Small client for LM Studio's local OpenAI-compatible image chat API."""

import base64
import io

import requests


class LMStudioClient:
    def __init__(self, server_url, timeout_seconds=600, session=None):
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def available_models(self):
        try:
            response = self.session.get(f"{self.server_url}/v1/models", timeout=10)
            response.raise_for_status()
            return sorted(item["id"] for item in response.json()["data"])
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach LM Studio at {self.server_url}. Start its local server first."
            ) from exc

    def model_metadata(self, model_id):
        """Best-effort metadata; older LM Studio versions may lack this route."""
        try:
            response = self.session.get(f"{self.server_url}/api/v1/models", timeout=10)
            response.raise_for_status()
            for item in response.json().get("models", []):
                instance_ids = [v.get("id") for v in item.get("loaded_instances", [])]
                if model_id == item.get("key") or model_id in instance_ids:
                    return item
        except (requests.RequestException, ValueError, KeyError):
            pass
        return None

    def unload_model(self, model_id):
        """Release only this model's loaded instances before CPU scoring."""
        metadata = self.model_metadata(model_id)
        if metadata is None:
            return 0  # Older LM Studio versions may not expose the native model list.
        for item in metadata.get("loaded_instances", []):
            instance_id = item.get("id")
            if not instance_id:
                continue
            response = self.session.post(
                f"{self.server_url}/api/v1/models/unload",
                json={"instance_id": instance_id}, timeout=60)
            response.raise_for_status()
        return len(metadata.get("loaded_instances", []))

    def answer(self, image, prompt, model_id, max_new_tokens, seed, reasoning_off=False):
        # Lossless PNG transport preserves the exact pixels of each prepared variant.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        if reasoning_off:
            # LM Studio's native API explicitly supports reasoning="off". The
            # OpenAI-compatible chat route has no documented equivalent.
            payload = {
                "model": model_id,
                "input": [
                    {"type": "image", "data_url": data_url},
                    {"type": "text", "content": prompt},
                ],
                "temperature": 0,
                "max_output_tokens": max_new_tokens,
                "reasoning": "off",
                "store": False,
                "stream": False,
            }
            response = self.session.post(
                f"{self.server_url}/api/v1/chat", json=payload,
                timeout=(10, self.timeout_seconds),
            )
            response.raise_for_status()
            body = response.json()
            content = "\n".join(
                item.get("content", "") for item in body.get("output", [])
                if item.get("type") == "message"
            ).strip()
            if not content:
                raise ValueError("LM Studio returned no final answer with reasoning=off.")
            return content

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0,
            "max_tokens": max_new_tokens,
            "seed": seed,
            "stream": False,
        }
        response = self.session.post(
            f"{self.server_url}/v1/chat/completions", json=payload,
            timeout=(10, self.timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"LM Studio returned no answer text: {body.get('choices', [])[:1]}")
        content = content.strip()
        if content.startswith("<think>"):
            if "</think>" not in content:
                raise ValueError("Model generated only a thinking block. Disable thinking in LM Studio.")
            content = content.split("</think>", 1)[1].strip()
        if not content:
            raise ValueError("Model returned no final answer. Disable thinking in LM Studio.")
        return content
