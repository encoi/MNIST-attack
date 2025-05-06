from models.wideresidual import WideResNet, WideBasic

def wideresnet28_10():
    """Return a WideResNet-28-10 model for CIFAR10"""
    return WideResNet(10, WideBasic, depth=28, widen_factor=10)

def wideresnet40_10():
    """Return a WideResNet-40-10 model for CIFAR10"""
    return WideResNet(10, WideBasic, depth=40, widen_factor=10)
