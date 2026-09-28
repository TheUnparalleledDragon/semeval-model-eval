import base64
import unittest

from PIL import Image

from local.lmstudio import LMStudioClient
from local.run_local import model_identity, selected_variants


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, content="A short answer"):
        self.last_payload = None
        self.content = content

    def get(self, url, timeout):
        if url.endswith("/api/v1/models"):
            return FakeResponse({"models": [{
                "key": "vision-model", "capabilities": {"vision": True},
                "quantization": {"name": "Q4_K_M"}, "size_bytes": 123,
            }]})
        if url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "vision-model"}]})
        raise AssertionError(url)

    def post(self, url, json, timeout):
        self.last_payload = json
        if url.endswith("/api/v1/chat"):
            return FakeResponse({"output": [{"type": "message", "content": "إجابة قصيرة"}],
                                 "stats": {"reasoning_output_tokens": 0}})
        return FakeResponse({"choices": [{"message": {"content": self.content}}]})


class LoadedSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.unloaded = []

    def get(self, url, timeout):
        if url.endswith("/api/v1/models"):
            return FakeResponse({"models": [
                {"key": "vision-model", "loaded_instances": [{"id": "vision-model:1"}]},
                {"key": "another-model", "loaded_instances": [{"id": "another-model:1"}]},
            ]})
        return super().get(url, timeout)

    def post(self, url, json, timeout):
        if url.endswith("/api/v1/models/unload"):
            self.unloaded.append(json["instance_id"])
            return FakeResponse({"instance_id": json["instance_id"]})
        return super().post(url, json, timeout)


class LocalRunnerTests(unittest.TestCase):
    def test_image_chat_payload_and_response(self):
        session = FakeSession()
        client = LMStudioClient("http://127.0.0.1:1234", session=session)
        self.assertEqual(client.available_models(), ["vision-model"])
        self.assertTrue(client.model_metadata("vision-model")["capabilities"]["vision"])
        result = client.answer(Image.new("RGB", (2, 2), "red"), "Question?",
                               "vision-model", 128, 42)
        self.assertEqual(result, "A short answer")
        payload = session.last_payload
        self.assertEqual(payload["messages"][0]["content"][1]["text"], "Question?")
        data_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(base64.b64decode(data_url.split(",", 1)[1]).startswith(b"\x89PNG"))
        self.assertEqual(payload["temperature"], 0)

    def test_reasoning_block_is_excluded(self):
        client = LMStudioClient("http://127.0.0.1:1234", session=FakeSession("<think>reason</think>Final"))
        self.assertEqual(client.answer(Image.new("RGB", (1, 1)), "Q", "vision-model", 128, 42), "Final")

    def test_native_request_disables_reasoning_and_reads_final_message(self):
        session = FakeSession(content="")
        client = LMStudioClient("http://127.0.0.1:1234", session=session)
        result = client.answer(Image.new("RGB", (1, 1)), "Q", "vision-model", 128, 42,
                               reasoning_off=True)
        self.assertEqual(result, "إجابة قصيرة")
        self.assertEqual(session.last_payload["reasoning"], "off")
        self.assertEqual(session.last_payload["max_output_tokens"], 128)
        self.assertEqual([item["type"] for item in session.last_payload["input"]],
                         ["image", "text"])

    def test_variants_and_stable_model_identity(self):
        self.assertEqual(selected_variants("train"), ("original",))
        self.assertEqual(selected_variants("dev"), ("original",))
        first = {"key": "m", "quantization": {"name": "Q4"}, "loaded_instances": [{"id": "x"}]}
        second = {**first, "loaded_instances": [{"id": "y"}]}
        self.assertEqual(model_identity("m", first), model_identity("m", second))

    def test_unloads_only_selected_model_before_scoring(self):
        session = LoadedSession()
        client = LMStudioClient("http://127.0.0.1:1234", session=session)
        self.assertEqual(client.unload_model("vision-model"), 1)
        self.assertEqual(session.unloaded, ["vision-model:1"])


if __name__ == "__main__":
    unittest.main()
