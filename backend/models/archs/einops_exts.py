from functools import wraps

from einops import repeat

# checking shape
# @nils-werner
# https://github.com/arogozhnikov/einops/issues/168#issuecomment-1042933838


# do same einops operations on a list of tensors


def _many(fn):

    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


# generate all helper functions

repeat_many = _many(repeat)
