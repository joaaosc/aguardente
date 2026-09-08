"""Stateless, named-tensor native validation for model task adapters."""
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

from .swift_runtime import SwiftRuntime


class TaskRuntime(SwiftRuntime):
    def __init__(self, asset, *, timeout=300):
        self.timeout = timeout
        self.closed = False
        self.temp = tempfile.TemporaryDirectory(prefix="aguardente-task-")
        self.root = Path(self.temp.name)
        self.timings = []
        self.host_peak_rss_bytes = 0
        try:
            executable = self.root / "runtime"
            subprocess.run(["xcrun", "swiftc", "-O", "-parse-as-library",
                            str(Path(__file__).with_suffix(".swift")), "-o", str(executable)],
                           check=True, timeout=120)
            self.process = subprocess.Popen([str(executable), str(Path(asset).resolve())],
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
            ready = json.loads(self._readline())
            self.function_names = ready["functions"]
            self.load_seconds = ready["load_seconds"]
        except BaseException:
            self.close()
            raise

    def predict(self, inputs, input_names, output_names):
        import numpy as np
        request = {"inputs": {}, "outputs": output_names, "directory": str(self.root)}
        for i, (name, tensor) in enumerate(zip(input_names, inputs, strict=True)):
            array = tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()
            if str(array.dtype) not in {"int32", "float16", "float32"}:
                raise ValueError(f"native task input dtype unsupported: {array.dtype}")
            path = self.root / f"input-{i}.bin"
            array.tofile(path)
            request["inputs"][name] = {"path": str(path), "dtype": str(array.dtype), "shape": list(array.shape)}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = json.loads(self._readline())
        self.timings.append(result["seconds"])
        self.host_peak_rss_bytes = max(self.host_peak_rss_bytes, result["host_peak_rss_bytes"])
        return tuple(np.fromfile(result["outputs"][n]["path"], dtype=np.float32)
                         .reshape(result["outputs"][n]["shape"]) for n in output_names)


def cmd_predict(args):
    import numpy as np
    from .errors import AguardenteError
    bundle = Path(args.bundle).expanduser().resolve()
    contract = json.loads((bundle / "interface.json").read_text())
    with np.load(args.inputs, allow_pickle=False) as data:
        names = contract["input_names"]
        if set(data.files) != set(names):
            raise AguardenteError("NPZ precisa ter exatamente as entradas de interface.json")
        inputs = tuple(data[name] for name in names)
    for array, shape, dtype in zip(inputs, contract["shapes"], contract["dtypes"], strict=True):
        if list(array.shape) != shape or str(array.dtype) != dtype.removeprefix("torch."):
            raise AguardenteError("formas/dtypes precisam corresponder a interface.json")
        if not np.isfinite(array).all():
            raise AguardenteError("entrada contém valores não finitos")
    output = Path(args.out).expanduser().resolve()
    if output == bundle or bundle in output.parents or output == Path(args.inputs).expanduser().resolve():
        raise AguardenteError("grave as previsões fora do bundle e do arquivo de entrada")
    runtime = TaskRuntime(bundle / "model.aimodel")
    try:
        values = runtime.predict(inputs, names, contract["output_names"])
        if not all(np.isfinite(v).all() for v in values):
            raise AguardenteError("runtime produziu valores não finitos")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            np.savez(file, **dict(zip(contract["output_names"], values, strict=True)))
        print(json.dumps({"outputs": str(output), "load_seconds": runtime.load_seconds,
                          "call_seconds": runtime.timings, "host_peak_rss_bytes": runtime.host_peak_rss_bytes}))
    finally:
        runtime.close()
    return 0
