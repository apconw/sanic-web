from glob import glob
import importlib.util
import inspect
import os
from pathlib import Path
from types import ModuleType
from typing import Union

from sanic.blueprints import Blueprint


def autodiscover(app, *module_names: Union[str, ModuleType], recursive: bool = False):
    """
    自动扫描目录添加蓝图/路由信息
    :param app:
    :param module_names:
    :param recursive:
    """
    mod = app.__module__
    blueprints = set()
    _imported = set()

    def _find_bps(module):
        nonlocal blueprints
        for _, member in inspect.getmembers(module):
            if isinstance(member, Blueprint):
                blueprints.add(member)

    for module in module_names:
        if isinstance(module, str):
            module = importlib.import_module(module, mod)
            _imported.add(module.__file__)
        _find_bps(module)

        if recursive:
            base = Path(module.__file__).parent
            # 使用os.path.join来构建正确的路径模式
            pattern = os.path.join(base, "**", "*.py")
            for path in glob(pattern, recursive=True):
                if path not in _imported:
                    name = "module"
                    if "__init__.py" in path:
                        *_, name, _ = path.split(os.sep)  # 使用os.sep作为路径分隔符
                    spec = importlib.util.spec_from_file_location(name, path)
                    speckled = importlib.util.module_from_spec(spec)
                    _imported.add(path)
                    spec.loader.exec_module(speckled)
                    _find_bps(speckled)

    for bp in blueprints:
        app.blueprint(bp)
