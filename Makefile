PROGRAM=PROG
BINDIR = bin
SRC = src
OBJ = obj
INC = include

CC     = nvcc
CLINK  = $(CC)
C_LIB  = -lm
CFLAGS = -std=c++11 -lineinfo -arch=sm_35 -O3 -I${INC} -x cu
CLINKFLAGS= -O3 -arch=sm_35

OBJS = ${OBJ}/_main_program.o \
	${OBJ}/initialize.o \
	${OBJ}/exact_solution.o \
	${OBJ}/exact_rhs.o \
	${OBJ}/set_constants.o \
	${OBJ}/adi.o \
	${OBJ}/compute_rhs.o \
	${OBJ}/x_solve.o \
	${OBJ}/y_solve.o \
	${OBJ}/z_solve.o \
	${OBJ}/error.o \
	${OBJ}/verify.o \
	${OBJ}/print_results.o \
	${OBJ}/timers.o

${BINDIR}/${PROGRAM}: ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${BINDIR}/${PROGRAM} ${OBJS} ${C_LIB}

${OBJ}/_main_program.o: ${SRC}/_main_program.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/_main_program.c -o ${OBJ}/_main_program.o
${OBJ}/initialize.o: ${SRC}/initialize.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/initialize.c -o ${OBJ}/initialize.o
${OBJ}/exact_solution.o: ${SRC}/exact_solution.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/exact_solution.c -o ${OBJ}/exact_solution.o
${OBJ}/exact_rhs.o: ${SRC}/exact_rhs.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/exact_rhs.c -o ${OBJ}/exact_rhs.o
${OBJ}/set_constants.o: ${SRC}/set_constants.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/set_constants.c -o ${OBJ}/set_constants.o
${OBJ}/adi.o: ${SRC}/adi.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/adi.c -o ${OBJ}/adi.o
${OBJ}/compute_rhs.o: ${SRC}/compute_rhs.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/compute_rhs.c -o ${OBJ}/compute_rhs.o
${OBJ}/x_solve.o: ${SRC}/x_solve.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/x_solve.c -o ${OBJ}/x_solve.o
${OBJ}/y_solve.o: ${SRC}/y_solve.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/y_solve.c -o ${OBJ}/y_solve.o
${OBJ}/z_solve.o: ${SRC}/z_solve.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/z_solve.c -o ${OBJ}/z_solve.o
${OBJ}/error.o: ${SRC}/error.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/error.c -o ${OBJ}/error.o
${OBJ}/verify.o: ${SRC}/verify.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/verify.c -o ${OBJ}/verify.o
${OBJ}/print_results.o: ${SRC}/print_results.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/print_results.c -o ${OBJ}/print_results.o
${OBJ}/timers.o: ${SRC}/timers.c ${INC}/header.h ${INC}/data_params.h
	$(CC) $(CFLAGS) -c ${SRC}/timers.c -o ${OBJ}/timers.o

clean:
	rm -f ${OBJ}/*.o
cleanall:
	rm -f ${OBJ}/*.o ${BINDIR}/${PROGRAM}
