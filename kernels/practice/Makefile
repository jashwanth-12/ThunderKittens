# Practice ladder. Override LEVEL on the command line:
#
#   make LEVEL=01 run     build + run
#   make LEVEL=01 ncu     Nsight Compute, full metric set -> level_01.out.ncu-rep
#   make LEVEL=01 nsys    Nsight Systems timeline        -> level_01.out.nsys-rep
#   make LEVEL=01 ptx     dump PTX, useful for checking what the compiler emitted
#   make LEVEL=01 clean
#
# Extra flags pass through, e.g.:
#   make LEVEL=01 NVCCFLAGS+=-DHARNESS_N=2048 run

LEVEL ?= 01

ARCH := SM90
SRC := level_$(LEVEL).cu
OUT := level_$(LEVEL).out
CMD := ./level_$(LEVEL).out
CONFIG := standalone
include ../common.mk
