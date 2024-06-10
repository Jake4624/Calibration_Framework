# make file for simple fortran

FORT = ifort
UMAT = Updated_Umat
#UMAT = umatNoCommonRatcheting2

mpDriverUmat: mpDriverUmat.o $(UMAT).o 
	$(FORT) -o  mpDriverUmat mpDriverUmat.o  $(UMAT).o -L/cm/shared/apps/lapack/ifort/3.9.0 -llapack -lblas

mpDriverUmat.o: mpDriverUmat.f90
	$(FORT) -c -g mpDriverUmat.f90

$(UMAT).o: $(UMAT).f
	$(FORT) -c -g $(UMAT).f

clean:
	rm mpDriverUmat mpDriverUmat.o $(UMAT).o
