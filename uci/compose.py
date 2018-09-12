import functools

def compose(functions):
    """Composes a bunch of functions"""
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)
