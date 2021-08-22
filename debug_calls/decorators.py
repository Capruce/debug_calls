import functools
from inspect import Parameter, signature, stack
from itertools import product
import sysconfig
from typing import Any, Callable, Sequence, Tuple, TypeVar
from unittest.util import safe_repr

CallableT = TypeVar("CallableT", bound=Callable[..., Any])

_STDLIB_PREFIX = sysconfig.get_path("stdlib")
_SITE_PACKAGES_PREFIX = sysconfig.get_path("purelib")

_FRAMES_TO_SKIP = (
    # Contextlib
    (f"{_STDLIB_PREFIX}/contextlib.py", "inner"),

    # Celery Tasks (i.e. `shared_task())`
    (f"{_SITE_PACKAGES_PREFIX}/celery/app/task.py", "__call__"),
    (f"{_SITE_PACKAGES_PREFIX}/celery/local.py", "__call__"),

    # DDTrace (i.e. `tracer.wrap()`)
    (f"{_SITE_PACKAGES_PREFIX}/ddtrace/tracer.py", "func_wrapper"),
)

_BOLD_CODE = "\033[1m"
_UNDERLINE_CODE = "\033[4m"
_END_CODE = "\033[0m"


def get_caller_info(frames_to_skip: Sequence[Tuple[str, str]] = _FRAMES_TO_SKIP) -> Tuple[str, int, str, bool]:
    """We often decorate our functions with useful wrapper functions like
    `@shared_task` . These intermediate functions appear in the call stack but
    are not so useful to debugging.
    """
    call_stack = stack()[2:]  # everything from the caller of the wrapped function.
    for i, (_, filename, lineno, function, _, _) in enumerate(call_stack):
        if (filename, function) in frames_to_skip:
            continue
        return filename, lineno, function, i > 0
    return "", 0, "", True


def debug_calls(__c: CallableT) -> CallableT:
    """
    >>> @debug_calls
    >>> def f(p_arg1, /, p_or_k_arg1, *, k_arg1, k_arg2="k_arg2"):
    >>>     pass
    >>> f("p_arg1", "p_or_k_arg1", k_arg1="k_arg1")

    Output:
    __main__.f
    Called by <module> at __main__.py:4
    (p) p_arg1     : "p_arg1"          (p means passed in by position)
    (p) p_or_k_arg1: "p_or_k_arg1"
    (k) k_arg1     : "k_arg1"          (k means passed in by keyword)
    (d) k_arg2     : "k_arg2"          (d means not passed in so default used)

    >>> @debug_calls
    >>> def f(p_arg1, *star_p_args, k_arg1, **star_k_args):
    >>>     pass
    >>> f(1, 2, 3, k_arg1=1, k_arg2=2, k_arg3=3)

    Output:
    __main__.f
    Called by <module> at __main__.py:4
    (p) p_arg1: 1
    (k) k_arg1: 1
     *star_p_args:
         (2, 3)
     **star_k_args:
         k_arg2: 2
         k_arg3: 3
    """
    parameters = signature(__c).parameters

    param_to_default = {}
    parg_names = []
    kwarg_names = []
    star_pargs_name = None
    star_kwargs_name = None

    for name, param in parameters.items():
        if param.kind is Parameter.POSITIONAL_ONLY:
            param_to_default[name] = param.default, "d"
            parg_names.append(name)
        elif param.kind is Parameter.POSITIONAL_OR_KEYWORD:
            param_to_default[name] = param.default, "d"
            parg_names.append(name)
            kwarg_names.append(name)
        elif param.kind is Parameter.KEYWORD_ONLY:
            param_to_default[name] = param.default, "d"
            kwarg_names.append(name)
        elif param.kind is Parameter.VAR_POSITIONAL:
            star_pargs_name = name
        elif param.kind is Parameter.VAR_KEYWORD:
            star_kwargs_name = name

    pad_length = max(len(name) for name in param_to_default)

    @functools.wraps(__c)
    def wrapper(*pargs, **kwargs):
        print()
        filename, lineno, caller_name, frames_skipped = get_caller_info()
        print(f"{_BOLD_CODE}{_UNDERLINE_CODE}{__c.__module__}.{__c.__qualname__}{_END_CODE}")
        print(f"Called by {caller_name} at {filename}:{lineno} {'(skipped decorators)' if frames_skipped else ''}")

        arg_to_value = param_to_default.copy()  # a copy keeping original insertion (i.e. definition) order.
        star_kwargs = {}

        # First apply all the kwargs
        for name, value in kwargs.items():
            if name in kwarg_names:
                arg_to_value[name] = value, "k"
            else:
                star_kwargs[name] = value

        # Of the list of parg_names, filter out those already used as kwargs
        remaining_parg_names = [name for name in parg_names if name not in kwargs.items()]

        arg_to_value.update(zip(remaining_parg_names, product(pargs, "p")))
        star_pargs = pargs[len(remaining_parg_names):]

        for name, (value, input_type) in arg_to_value.items():
            print(f"({input_type}) {name:<{pad_length}}: {safe_repr(value, short=True)}")
        if star_pargs_name is not None:
            print(f" *{star_pargs_name}:")
            print(f"     {star_pargs}")
        if star_kwargs_name is not None:
            print(f" **{star_kwargs_name}:")
            star_kwarg_pad_length = max(len(name) for name in star_kwargs)
            for name, value in star_kwargs.items():
                print(f"     {name:<{star_kwarg_pad_length}}: {value!r}")
        return __c(*pargs, **kwargs)

    # Ok, yes, *technically* `f(*args, **kwargs) -> Any` is not the same type
    # as `CallableT` buuuuuut it calls directly down into it so it effectively
    # has the same signature at runtime.
    # We can actually encode this logic once PEP 612 is backported.
    return wrapper  # type: ignore[return-value]
