# Unified Makefile: ncurses demos (linear + polynomial regression)
# Files expected in this directory:
#   - linreg_ncurses.c
#   - polyreg_ncurses.c

# Toolchain
CC      ?= gcc

# Common flags
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
LDLIBS  ?= -lncurses -lm

# Stupid warning
CFLAGS += -D_XOPEN_SOURCE=700
# (you can use -D_POSIX_C_SOURCE=200809L instead if you prefer)


# Targets
LIN_TGT  ?= linreg_ncurses
LIN_SRC  ?= linreg_ncurses.c
LIN_OBJ   = $(LIN_SRC:.c=.o)

POLY_TGT ?= polyreg_ncurses
POLY_SRC ?= polyreg_ncurses.c
POLY_OBJ  = $(POLY_SRC:.c=.o)

.PHONY: all linreg polyreg run_linreg run_polyreg debug_linreg debug_polyreg clean

# Build both by default
all: linreg polyreg

# --- Linear regression ---
linreg: $(LIN_TGT)
$(LIN_TGT): $(LIN_OBJ)
	$(CC) $(LIN_OBJ) -o $@ $(LDLIBS)
$(LIN_OBJ): $(LIN_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

run_linreg: $(LIN_TGT)
	./$(LIN_TGT)

debug_linreg: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug_linreg: LDLIBS := -lncurses -lm -fsanitize=address,undefined
debug_linreg: clean linreg

# --- Polynomial regression ---
polyreg: $(POLY_TGT)
$(POLY_TGT): $(POLY_OBJ)
	$(CC) $(POLY_OBJ) -o $@ $(LDLIBS)
$(POLY_OBJ): $(POLY_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

run_polyreg: $(POLY_TGT)
	./$(POLY_TGT)

debug_polyreg: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug_polyreg: LDLIBS := -lncurses -lm -fsanitize=address,undefined
debug_polyreg: clean polyreg

# --- Housekeeping ---
clean:
	rm -f $(LIN_OBJ) $(POLY_OBJ) $(LIN_TGT) $(POLY_TGT)

