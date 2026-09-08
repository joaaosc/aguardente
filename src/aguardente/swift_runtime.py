"""Use the OS Core AI API with the dynamic-shape specialization contract."""
import json
from pathlib import Path
import subprocess
import tempfile
import queue
import threading


class SwiftRuntime:
    def __init__(self, asset, shape, precision, *, timeout=300):
        self.timeout = timeout
        self.closed = False
        self.temp = tempfile.TemporaryDirectory(prefix="aguardente-runtime-")
        self.root = Path(self.temp.name)
        executable = self.root / "runtime"
        try:
            subprocess.run(["xcrun", "swiftc", "-O", "-parse-as-library",
                            str(Path(__file__).with_name("runtime_bridge.swift")),
                            "-o", str(executable)], check=True, timeout=120)
            self.process = subprocess.Popen([str(executable), str(Path(asset).resolve()),
                                             json.dumps(shape), precision],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.lines = queue.Queue()
            def reader():
                try:
                    for line in self.process.stdout:
                        self.lines.put(line)
                finally:
                    self.lines.put(None)
            self.reader = threading.Thread(target=reader, daemon=True)
            self.reader.start()
            self.function_names = json.loads(self._readline())["functions"]
        except BaseException:
            self.close()
            raise
        self.counter = 0

    def _readline(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.process.kill()
            self.process.wait()
            raise TimeoutError(f"Core AI did not respond within {self.timeout}s") from None
        if line is None:
            raise RuntimeError(f"Core AI Swift runtime exited ({self.process.poll()})")
        return line

    def new_state(self):
        self.counter += 1
        return self.counter

    async def call(self, function, ids, total, state):
        import numpy as np
        path = self.root / "logits.bin"
        request = dict(function=function, ids=ids, total=total, state=state, output=str(path))
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self._readline()
        result = json.loads(line)
        if "shape" not in result:
            return {}
        array = np.fromfile(path, dtype=np.float32).reshape(result["shape"])

        class Output:
            def numpy(self):
                return array
        return {"logits": Output()}

    async def release_state(self, state):
        await self.call("__release__", [], 0, state)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "process"):
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            if hasattr(self, "reader"):
                self.reader.join(timeout=2)
            self.process.stdout.close()
        self.temp.cleanup()
