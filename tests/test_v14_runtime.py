import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).parents[1]/'scripts/a3_v14_runtime.py'
    spec=importlib.util.spec_from_file_location('v14_runtime',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('failure',['interrupt','identity'])
def test_supervisor_setup_or_interrupt_cleans_owned_child(tmp_path,monkeypatch,failure):
    m=module()
    monkeypatch.setattr(m,'verify',lambda root:dict(jobs={'stage1':dict(output=str(tmp_path/'output'))}))
    class Child:
        pid=123456
        def __init__(self):self.returncode=None;self.terminated=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-1
        def wait(self,timeout=None):return self.returncode
    child=Child();watchdog=Child();queue=[child,watchdog]
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:queue.pop(0))
    def identity(process):
        if failure=='identity' and process==123456:raise RuntimeError('setup identity failed')
        return dict(pid=123456,created=1,exe='fixture',cmdline=[])
    monkeypatch.setattr(m,'identity',identity)
    monkeypatch.setattr(m.psutil,'Process',lambda pid=None:pid)
    monkeypatch.setattr(m,'terminate_owned',lambda saved:False)
    def interrupt(**kwargs):raise KeyboardInterrupt('fixture interrupt')
    monkeypatch.setattr(m,'guard_reason',interrupt)
    expected=RuntimeError if failure=='identity' else KeyboardInterrupt
    with pytest.raises(expected):m.supervise(tmp_path,'stage1')
    assert child.terminated
    assert (tmp_path/'sessions/stage1/status.json').exists()
    assert (tmp_path/'sessions/stage1/watchdog_stop.json').exists()


@pytest.mark.parametrize('benchmark', [False, True])
def test_child_witness_records_actual_backend_after_runner(tmp_path, monkeypatch, benchmark):
    import json
    import torch
    import a3_ultrasound.runtime_supervisor as supervisor
    import a3_ultrasound.stream_training as runner
    from a3_ultrasound.speed import SpeedOptions, configure_backend
    m = module()
    out = tmp_path / 'output'; out.mkdir()
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    monkeypatch.setattr(m, 'verify', lambda root: dict(protocol='V1.5-freeze', jobs={'job': dict(output=str(out), head='proposed_v15')}))
    monkeypatch.setattr(m, 'read', lambda path: {})
    monkeypatch.setattr(m, 'matching_process', lambda value: object())
    monkeypatch.setattr(supervisor, 'install_atomic_outputs', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'fixture 3080 Ti')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *args: type('Props', (), dict(total_memory=12*1024**3))())
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 123)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 456)
    monkeypatch.setattr(runner, 'train', lambda *args, **kwargs: configure_backend(SpeedOptions(tf32=True, cudnn_benchmark=benchmark)))
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled())
    try:
        m.child(tmp_path, 'job')
        witness = json.loads((out / 'execution_witness.json').read_text())
        assert witness['tf32'] is witness['tf32_cudnn'] is True
        assert witness['deterministic'] is witness['cudnn_deterministic'] is (not benchmark)
        assert witness['cudnn_benchmark'] is benchmark
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original[:2]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original[2:4]
        torch.use_deterministic_algorithms(original[4])
