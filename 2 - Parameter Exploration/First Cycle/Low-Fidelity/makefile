# make file for simple fortran

FORT = ifort
UMAT = Updated_Umat

#MPDRIVER = mpDriverUmat
MPDRIVER = mpDriverUmat_DFGRD

$(MPDRIVER): $(MPDRIVER).o $(UMAT).o 
	$(FORT) -o  $(MPDRIVER) $(MPDRIVER).o  $(UMAT).o -L/cm/shared/apps/lapack/ifort/3.9.0 -llapack -lblas

$(MPDRIVER).o: $(MPDRIVER).f90
	$(FORT) -c -g $(MPDRIVER).f90

$(UMAT).o: $(UMAT).f
	$(FORT) -c -g $(UMAT).f

clean:
	rm $(MPDRIVER) $(MPDRIVER).o $(UMAT).o
