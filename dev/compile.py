import functools
import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config

def compile_with_debug(fn, compile_kwargs=None, dynamo_kwargs=None, inductor_kwargs=None):
    """Wrap a function with torch.compile and enable debug output.

    Debug config is applied when compilation actually occurs (on first call).

    Args:
        fn: The function to compile.
        compile_kwargs: Dict of arguments passed to torch.compile (e.g. mode, fullgraph, dynamic).
        inductor_kwargs: Dict of extra inductor config overrides merged with debug defaults.
    """
    if compile_kwargs is None:
        compile_kwargs = {}
    if inductor_kwargs is None:
        inductor_kwargs = {}
    if dynamo_kwargs is None:
        dynamo_kwargs = {}

    if "mode" in compile_kwargs:
        mode_options = torch._inductor.list_mode_options(compile_kwargs.pop("mode"))
        inductor_kwargs = {**mode_options, **inductor_kwargs}

    compiled_fn = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal compiled_fn

        if compiled_fn is None:
            # First call - compile with debug settings and caching disabled
            with dynamo_config.patch(
                     verbose=True,
                     **dynamo_kwargs,
                 ), \
                 inductor_config.patch(
                    **inductor_kwargs, **{
                     "debug": True,
                     "trace.enabled": True,
                     "trace.fx_graph": True,
                     "trace.fx_graph_transformed": True,
                     "trace.output_code": True,
                     "fx_graph_cache": False,
                     "force_disable_caches": True
                 }):
                compiled_fn = torch.compile(fn, options=inductor_kwargs, **compile_kwargs)
                return compiled_fn(*args, **kwargs)

        # No recompilation flag. The function was already compiled with dynamo and inductor configs.
        # This early return avoids inductor context manager overhead.
        if "cache_size_limit" in dynamo_kwargs and dynamo_kwargs["cache_size_limit"] == 1:
            # We add error_on_recompile to avoid silent eager fallbacks
            with dynamo_config.patch(**dynamo_kwargs, error_on_recompile=True):
                return compiled_fn(*args, **kwargs)

        # Apply config on every call so recompilations
        # (triggered by dynamo guard failures) still use cpp_wrapper, capture_scalar_outputs, etc.
        with dynamo_config.patch(**dynamo_kwargs):
            #  inductor_config.patch(**inductor_kwargs):
            return compiled_fn(*args, **kwargs)

    return wrapper

def warmup_compiled_fn(compiled_fn, *args, **kwargs):
    for _ in range(10):
        compiled_fn(*args, **kwargs)