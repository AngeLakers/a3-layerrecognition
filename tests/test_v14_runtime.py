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
