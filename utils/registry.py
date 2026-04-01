"""
Registry 系统 - 用于统一管理模型、数据集、损失函数等组件
"""


class Registry:
    """
    简单的注册器类，用于将字符串名称映射到实际的类或函数。
    支持注册和获取操作。
    """

    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    def register(self, obj=None, name: str = None):
        """
        注册一个对象到注册器。

        Args:
            obj: 要注册的对象（类或函数）
            name: 注册名称，默认使用对象的 __name__ 属性

        Returns:
            如果 obj 为 None，返回装饰器函数；否则返回注册的对象
        """
        if obj is None:
            # 作为装饰器使用
            def decorator(func_or_class):
                self.register(func_or_class, name)
                return func_or_class
            return decorator

        # 使用提供的名称或对象的名称
        reg_name = name if name is not None else obj.__name__
        if reg_name in self._obj_map:
            raise KeyError(f"{self._name} registry: '{reg_name}' already registered!")

        self._obj_map[reg_name] = obj
        return obj

    def get(self, name: str):
        """
        根据名称获取注册的对象。

        Args:
            name: 注册时的名称

        Returns:
            注册的对象

        Raises:
            KeyError: 如果名称未注册
        """
        if name not in self._obj_map:
            raise KeyError(f"{self._name} registry: '{name}' not found. "
                          f"Available: {list(self._obj_map.keys())}")
        return self._obj_map[name]

    def list_available(self):
        """列出所有已注册的名称"""
        return list(self._obj_map.keys())


# 创建全局注册器实例
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
LOSS_REGISTRY = Registry("loss")
