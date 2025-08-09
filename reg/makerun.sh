# build both
make

# run either
make run_linreg
make run_polyreg

# debug builds with sanitizers
make debug_linreg
make debug_polyreg

# tweak dataset size / max degree (compile-time)
make CFLAGS+='-DNPTS=300 -DMAX_DEG=12'        # both targets
make polyreg CFLAGS+='-DNPTS=200 -DMAX_DEG=8' # just polyreg

