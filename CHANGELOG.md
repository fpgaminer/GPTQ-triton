# Changelog

## [0.0.3] - 2023-04-19

- Added support for groupsize.
    - Note: fuse_mlp is not recommended for groupsize != -1.  It is now disabled automatically during loading if the model has grouping, unless fuse_mlp is explictly set to True.  This is a result of the current kernel implementation being slower than the naive implementation for groupsize != -1.
- Added a warning if `act_order` and `groupsize` are used together.  They are not compatible.