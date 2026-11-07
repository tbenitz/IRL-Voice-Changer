"""Static regression checks for real Pocket ONNX browser model testing.

These checks do not claim audio inference works on a particular Chromebook.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "pocket-demo" / "studio"


class StudioSmokeTests(unittest.TestCase):
    def test_model_loader_controls_and_navigation(self):
        html = (STUDIO / "index.html").read_text(encoding="utf-8")
        for ident in ("btn-load-model", "btn-retry-model", "runtime-profile",
                      "runtime-detail", "model-progress", "model-ready-indicator"):
            self.assertIn(f'id="{ident}"', html)
        self.assertIn('href="../../online-test/"', html)
        self.assertRegex(html, r'src="app[.]js[?]v=[^"]+"')
        self.assertIn('trainer-addon.js?', html)
        self.assertIn('studio-addon.js?', html)
        # Cache-busted main Pocket controller must include silent batch handling.
        app = (STUDIO / "app.js").read_text(encoding="utf-8")
        self.assertIn('this.silentBatchMode = false', app)
        self.assertIn('if (!this.silentBatchMode) this.player.playAudio(pcm)', app)

    def test_workload_is_opt_in_and_uses_real_onnx(self):
        app = (STUDIO / "app.js").read_text(encoding="utf-8")
        worker = (STUDIO / "inference-worker.js").read_text(encoding="utf-8")
        self.assertIn('this.btnLoadModel.addEventListener("click"', app)
        self.assertIn('this.worker.postMessage({ type: "load", data: { profile:', app)
        self.assertNotIn('this.worker.postMessage({ type: "load" });', app)
        self.assertIn('ort.InferenceSession.create(bundlePath(language, stem)', worker)
        self.assertIn('for (const [i, [label, stem]] of stages.entries())', worker)
        self.assertNotIn('const results = await Promise.all([', worker)
        self.assertIn('const ortScript = "ort.min.js"', worker)
        self.assertIn('enableCpuMemArena: !lowMemoryMode', worker)

    def test_model_comparison_starts_locally(self):
        html = (ROOT / "online-test" / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="../live-vc/#rvc"', html)
        self.assertIn('href="../live-vc/#live"', html)
        self.assertIn('RVC ONNX', html)
        self.assertNotIn('<iframe', html)
        self.assertNotIn('hf.space', html)

    def test_microphone_transcription_disclosure(self):
        html = (STUDIO / "index.html").read_text(encoding="utf-8")
        self.assertIn("Browser speech recognition may use an online service", html)
        self.assertIn("low memory", html)


if __name__ == "__main__":
    unittest.main()
